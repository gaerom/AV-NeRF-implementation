import os 
import numpy as np
import pyroomacoustics
from scipy.signal import hilbert
import librosa
import argparse
from math import pi
from numpy import linalg as LA
import einops

import torch
import torch.nn.functional as F
import torchaudio
import torch.nn as nn


################################################### NeRAF #########################################################
class SpectralLoss(nn.Module):
    """
    Compute a loss between two log power-spectrograms. 
    From  https://github.com/facebookresearch/SING/blob/main/sing/dsp.py#L79 modified

    Arguments:
        base_loss (function): loss used to compare the log power-spectrograms.
            For instance :func:`F.mse_loss`
        epsilon (float): offset for the log, i.e. `log(epsilon + ...)`
        **kwargs (dict): see :class:`STFT`
    """

    def __init__(self, base_loss=F.mse_loss, reduction='mean', epsilon=1, dB=False, stft_input_type='mag', **kwargs):
        super(SpectralLoss, self).__init__()
        self.base_loss = base_loss
        self.epsilon = epsilon
        self.dB = dB
        self.stft_input_type = stft_input_type
        self.reduction = reduction

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def _log_spectrogram(self, STFT):
        if self.dB and self.stft_input_type == 'mag':
            return 10*torch.log10(self.epsilon + STFT) 
        elif not self.dB and self.stft_input_type == 'mag':
            return torch.log(self.epsilon + STFT)
        elif self.stft_input_type == 'log mag': 
            return STFT

    def forward(self, a, b):
        spec_a = self._log_spectrogram(a)
        spec_b = self._log_spectrogram(b)
        return self.base_loss(spec_a, spec_b, reduction=self.reduction)
    

def compute_t60(true_in, gen_in, fs, advanced = False):
    ch = true_in.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        try:
            if advanced: 
                true = measure_rt60_advance(true_in[c], sr=fs)
                gen = measure_rt60_advance(gen_in[c], sr=fs)
            else:
                true = pyroomacoustics.experimental.measure_rt60(true_in[c], fs=fs, decay_db=30)
                gen = pyroomacoustics.experimental.measure_rt60(gen_in[c], fs=fs, decay_db=30)
        except:
            true = -1
            gen = -1
        gt.append(true)
        pred.append(gen)
    return np.array(gt), np.array(pred)

def measure_rt60_advance(signal, sr, decay_db=10, cutoff_freq=200):
    # following RAF implementation
    signal = torch.from_numpy(signal)
    signal = torchaudio.functional.highpass_biquad(
        waveform=signal,
        sample_rate=sr,
        cutoff_freq=cutoff_freq
    )
    signal = signal.cpu().numpy()
    rt60 = pyroomacoustics.experimental.measure_rt60(signal, sr, decay_db=decay_db, plot=False)
    return rt60

def measure_clarity(signal, time=50, fs=44100):
    h2 = signal**2
    t = int((time/1000)*fs + 1) 
    return 10*np.log10(np.sum(h2[:t])/np.sum(h2[t:]))

def evaluate_clarity(pred_ir, gt_ir, fs):
    np_pred_ir = pred_ir
    np_gt_ir = gt_ir

    # manage multiple channels IR
    ch = gt_ir.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        pred_clarity = measure_clarity(np_pred_ir[c,...], fs=fs)
        gt_clarity = measure_clarity(np_gt_ir[c,...], fs=fs)
        gt.append(gt_clarity)
        pred.append(pred_clarity)
    return np.array(gt), np.array(pred)

def measure_edt(h, fs=44100, decay_db=10):
    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    if np.all(energy == 0):
        return np.nan

    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    i_decay = np.min(np.where(- decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs
    # compute the decay time
    decay_time = t_decay
    est_edt = (60 / decay_db) * decay_time 
    return est_edt

def evaluate_edt(pred_ir, gt_ir, fs):
    np_pred_ir = pred_ir
    np_gt_ir = gt_ir

    # manage multiple channels IR
    ch = gt_ir.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        pred_edt = measure_edt(np_pred_ir[c], fs=fs)
        gt_edt = measure_edt(np_gt_ir[c], fs=fs)
        gt.append(gt_edt)
        pred.append(pred_edt)
    return np.array(gt), np.array(pred)
    
################################################### NeRAF #########################################################

def energy_decay(pred_mag, gt_mag):
    # [B, T, F]
    gt_mag = torch.sum(gt_mag ** 2, dim=2)
    pred_mag = torch.sum(pred_mag ** 2, dim=2)
    gt_mag = torch.log1p(gt_mag)
    pred_mag = torch.log1p(pred_mag)

    loss = F.l1_loss(gt_mag, pred_mag)
    return loss

'''
def energy_decay(pred_mag, gt_mag):
    pred_mag = einops.rearrange(pred_mag, "b t f -> b f t")
    gt_mag = einops.rearrange(gt_mag, "b t f -> b f t")
    # [B, F, T]

    gts_fullBandAmpEnv = torch.sum(gt_mag, dim=1)
    power_gts_fullBandAmpEnv = gts_fullBandAmpEnv ** 2
    energy_gts_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_gts_fullBandAmpEnv, [1]), 1), [1])
    valid_loss_idxs = ((energy_gts_fullBandAmpEnv != 0.).type(energy_gts_fullBandAmpEnv.dtype))[:, 1:]
    db_gts_fullBandAmpEnv = 10 * torch.log10(energy_gts_fullBandAmpEnv + 1.0e-13)
    norm_db_gts_fullBandAmpEnv = db_gts_fullBandAmpEnv - db_gts_fullBandAmpEnv[:, :1]
    norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv[:, 1:]
    weighted_norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv * valid_loss_idxs

    preds_fullBandAmpEnv = torch.sum(pred_mag, dim=1)
    power_preds_fullBandAmpEnv = preds_fullBandAmpEnv ** 2
    energy_preds_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_preds_fullBandAmpEnv, [1]), 1), [1])
    db_preds_fullBandAmpEnv = 10 * torch.log10(energy_preds_fullBandAmpEnv + 1.0e-13)
    norm_db_preds_fullBandAmpEnv = db_preds_fullBandAmpEnv - db_preds_fullBandAmpEnv[:, :1]
    norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv[:, 1:]
    weighted_norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv * valid_loss_idxs

    loss = F.l1_loss(weighted_norm_db_preds_fullBandAmpEnv, weighted_norm_db_gts_fullBandAmpEnv)
    return loss
'''

def istft(mag, phase):
    mag = mag.cpu().numpy()
    phase = phase.cpu().numpy()
    spec = mag * np.exp(1j * phase)
    if spec.ndim == 2:
        spec = spec.T
    elif spec.ndim == 3:
        spec = einops.rearrange(spec, "c t f -> c f t")
    else:
        raise NotImplementedError
    wav = librosa.istft(spec, n_fft=512)
    if wav.ndim == 2:
        wav = wav.T
    return wav

class Evaluator(object):
    def __init__(self, norm=False):
        self.env_loss = []
        self.mag_loss = []
        self.snr_loss = []
        self.snr_norm_loss = []
        self.norm = norm
    
    def update(self, mag_prd, mag_gt, wav_prd, wav_gt):
        mag_loss = np.mean(np.power(mag_prd - mag_gt, 2)) * 2
        self.mag_loss.append(mag_loss)
        env_loss = self.Envelope_distance(wav_prd, wav_gt)
        self.env_loss.append(env_loss)
        snr_loss = self.SNR(wav_prd, wav_gt)
        self.snr_loss.append(snr_loss)

        wav_prd = normalize(wav_prd)
        wav_gt = normalize(wav_gt)
        snr_norm_loss = self.SNR(wav_prd, wav_gt)
        self.snr_norm_loss.append(snr_norm_loss)
        
        return [mag_loss, env_loss, snr_loss, snr_norm_loss]

    def report(self):
        item_len = len(self.mag_loss)
        return {
                "env": sum(self.env_loss) / item_len,
                "mag": sum(self.mag_loss) / item_len,
                "snr": sum(self.snr_loss) / item_len,
                "snr_norm": sum(self.snr_norm_loss) / item_len
                }
    
    def STFT_L2_distance(self, predicted_binaural, gt_binaural):
        #channel1
        predicted_spect_channel1 = librosa.stft(predicted_binaural[0,:], n_fft=512)
        gt_spect_channel1 = librosa.stft(gt_binaural[0,:], n_fft=512)
        real = np.expand_dims(np.real(predicted_spect_channel1), axis=0)
        imag = np.expand_dims(np.imag(predicted_spect_channel1), axis=0)
        predicted_realimag_channel1 = np.concatenate((real, imag), axis=0)
        real = np.expand_dims(np.real(gt_spect_channel1), axis=0)
        imag = np.expand_dims(np.imag(gt_spect_channel1), axis=0)
        gt_realimag_channel1 = np.concatenate((real, imag), axis=0)
        channel1_distance = np.mean(np.power((predicted_realimag_channel1 - gt_realimag_channel1), 2))

        #channel2
        predicted_spect_channel2 = librosa.stft(predicted_binaural[1,:], n_fft=512)
        gt_spect_channel2 = librosa.stft(gt_binaural[1,:], n_fft=512)
        real = np.expand_dims(np.real(predicted_spect_channel2), axis=0)
        imag = np.expand_dims(np.imag(predicted_spect_channel2), axis=0)
        predicted_realimag_channel2 = np.concatenate((real, imag), axis=0)
        real = np.expand_dims(np.real(gt_spect_channel2), axis=0)
        imag = np.expand_dims(np.imag(gt_spect_channel2), axis=0)
        gt_realimag_channel2 = np.concatenate((real, imag), axis=0)
        channel2_distance = np.mean(np.power((predicted_realimag_channel2 - gt_realimag_channel2), 2))

        #sum the distance between two channels
        stft_l2_distance = channel1_distance + channel2_distance
        return float(stft_l2_distance)

    def Envelope_distance(self, predicted_binaural, gt_binaural):
        #channel1
        pred_env_channel1 = np.abs(hilbert(predicted_binaural[0,:]))
        gt_env_channel1 = np.abs(hilbert(gt_binaural[0,:]))
        channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))
    
        #channel2
        pred_env_channel2 = np.abs(hilbert(predicted_binaural[1,:]))
        gt_env_channel2 = np.abs(hilbert(gt_binaural[1,:]))
        channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))
    
        #sum the distance between two channels
        envelope_distance = channel1_distance + channel2_distance
        return float(envelope_distance)

    def SNR(self, predicted_binaural, gt_binaural):
        mse_distance = np.mean(np.power((predicted_binaural - gt_binaural), 2))
        snr = 10. * np.log10((np.mean(gt_binaural**2) + 1e-4) / (mse_distance + 1e-4))

        return float(snr)
    
    def Magnitude_distance(self, predicted_binaural, gt_binaural):
        predicted_spect_channel1 = librosa.stft(predicted_binaural[0,:], n_fft=512)
        gt_spect_channel1 = librosa.stft(gt_binaural[0,:], n_fft=512)
        predicted_spect_channel2 = librosa.stft(predicted_binaural[1,:], n_fft=512)
        gt_spect_channel2 = librosa.stft(gt_binaural[1,:], n_fft=512)
        stft_mse1 = np.mean(np.power(np.abs(predicted_spect_channel1) - np.abs(gt_spect_channel1), 2))
        stft_mse2 = np.mean(np.power(np.abs(predicted_spect_channel2) - np.abs(gt_spect_channel2), 2))

        return float(stft_mse1 + stft_mse2)

    def Angle_Diff_distance(self, predicted_binaural, gt_binaural):
        gt_diff = gt_binaural[0] - gt_binaural[1]
        pred_diff = predicted_binaural[0] - predicted_binaural[1]
        gt_diff_spec = librosa.stft(gt_diff, n_fft=512)
        pred_diff_spec = librosa.stft(pred_diff, n_fft=512)
        _, pred_diff_phase = librosa.magphase(pred_diff_spec)
        _, gt_diff_phase = librosa.magphase(gt_diff_spec)
        pred_diff_angle = np.angle(pred_diff_phase)
        gt_diff_angle = np.angle(gt_diff_phase)
        angle_diff_init_distance = np.abs(pred_diff_angle - gt_diff_angle)
        angle_diff_distance = np.mean(np.minimum(angle_diff_init_distance, np.clip(2 * pi - angle_diff_init_distance, a_min=0, a_max=2*pi))) 

        return float(angle_diff_distance)

def normalize(samples):
    return samples / np.maximum(1e-20, np.max(np.abs(samples)))

def myhibert(x, axis=1):
    # Make input a real tensor
    x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    x = x.to(dtype=torch.float)

    if (axis < 0) or (axis > len(x.shape) - 1):
        raise ValueError(f"Invalid axis for shape of x, got axis {axis} and shape {x.shape}.")

    n = x.shape[axis]
    if n <= 0:
        raise ValueError("N must be positive.")
    x = torch.as_tensor(x, dtype=torch.complex64)
    # Create frequency axis
    f = torch.cat(
        [
            torch.true_divide(torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)),
            torch.true_divide(torch.arange(-(n // 2), 0, device=x.device), float(n)),
        ]
    )
    xf = torch.fft.fft(x, n=n, dim=axis)
    # Create step function
    u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
    u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
    new_dims_before = axis
    new_dims_after = len(xf.shape) - axis - 1
    for _ in range(new_dims_before):
        u.unsqueeze_(0)
    for _ in range(new_dims_after):
        u.unsqueeze_(-1)

    ht = torch.fft.ifft(xf * 2 * u, dim=axis)

    # Apply transform
    return torch.as_tensor(ht, device=ht.device, dtype=ht.dtype)


""" Sound Spaces Evaluator """
class SoundSpacesEvaluator(object):
    def __init__(self, fs=22050):
        self.fs = fs # sample rate
    
    def get_full_metrics(self, mag_prd, mag_gt, wav_gt_ff, wav_pred_istft, wav_gt_istft,  log_prd, log_gt):#, gl=False):
        
        wav_prd = wav_pred_istft
        wav_gt = wav_gt_istft
        n_ch = wav_gt.shape[0]

        # Zero pad waveform to be the same size as gt ff (i.e., max_len * hop_len )
        wav_prd = np.pad(wav_prd, ((0,0),(0, wav_gt_ff.shape[1]-wav_prd.shape[1])), 'constant', constant_values=(0,0))
        wav_gt = np.pad(wav_gt, ((0,0),(0, wav_gt_ff.shape[1]-wav_gt.shape[1])), 'constant', constant_values=(0,0))

        ## Waveform related
        # Compute t60 error, edt and c50 on gt from file
        t60s_gt, t60s_prd = compute_t60(wav_gt_ff, wav_prd, fs=self.fs) # utils.py
        t60s = np.concatenate((t60s_gt, t60s_prd))
        t60s = np.expand_dims(t60s, axis=0)
        diff = np.abs(t60s[:,n_ch:]-t60s[:,:n_ch])/np.abs(t60s[:,:n_ch])
        mask = np.any(t60s<-0.5, axis=1)
        diff = np.mean(diff, axis=1)
        diff[mask] = 1
        mean_t60error = np.mean(diff)*100
        invalid = np.sum(mask)
        
        edt_gt, edt_prd = evaluate_edt(wav_prd, wav_gt_ff, fs=self.fs)
        edts = np.concatenate((edt_gt, edt_prd))
        edt_instance = np.abs(edts[n_ch:]-edts[:n_ch]) # pred-gt
        mean_edt = np.mean(edt_instance, axis=0) # mean over instance channels

        c50_gt, c50_prd = evaluate_clarity(wav_prd, wav_gt_ff, fs=self.fs)
        c50s = np.concatenate((c50_gt, c50_prd))
        c50_instance = np.abs(c50s[n_ch:]-c50s[:n_ch]) # pred-gt
        mean_c50 = np.mean(c50_instance, axis=0) # mean over instance channels

        res = {
                "audio_T60_mean_error": mean_t60error,
                "audio_total_invalids_T60": invalid,
                "audio_EDT": mean_edt,
                "audio_C50": mean_c50,
                }
        
        for key in res.keys():
            #if tensor go to numpy
            if isinstance(res[key], torch.Tensor):
                res[key] = res[key].item()
            else:
                res[key] = float(res[key])

        return res
        
    
    def get_stft_metrics(self,mag_prd,mag_gt):
        ## STFT related
        mag_loss = torch.mean(torch.pow(mag_prd - mag_gt, 2)) * 2

        return {
                "audio_mag": mag_loss,
        }