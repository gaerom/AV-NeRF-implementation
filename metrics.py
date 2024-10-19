import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import hilbert
from numpy import pi


def energy_decay(pred_mag, gt_mag):
    
    gt_mag = torch.sum(gt_mag ** 2, dim=2)
    pred_mag = torch.sum(pred_mag ** 2, dim=2)
    gt_mag = torch.log1p(gt_mag)
    pred_mag = torch.log1p(pred_mag)
    
    loss = F.l1_loss(gt_mag, pred_mag)
    return loss


def Magnitude_distance(pred_wav, gt_wav):

    pred_stft = librosa.stft(pred_wav, n_fft=512)
    gt_stft = librosa.stft(gt_wav, n_fft=512)

    # Magnitude (절대값) 계산
    pred_mag = np.abs(pred_stft)
    gt_mag = np.abs(gt_stft)

    # Magnitude 차이에 대한 MSE 계산
    mag_loss = np.mean(np.power(pred_mag - gt_mag, 2)) * 2
    return mag_loss


def Envelope_distance(pred_wav, gt_wav):

    def extract_envelope(wav):
        return np.abs(hilbert(wav))

    pred_env = extract_envelope(pred_wav)
    gt_env = extract_envelope(gt_wav)

    env_loss = np.sqrt(np.mean(np.power(gt_env - pred_env, 2)))
    return env_loss


def calculate(pred_wav, gt_wav):

    mag_loss = Magnitude_distance(pred_wav, gt_wav)
    env_loss = Envelope_distance(pred_wav, gt_wav)


    return {
        "mag_loss": mag_loss,
        "env_loss": env_loss
    }


def evaluate(pred_wav, gt_wav):
   
    metrics = calculate(pred_wav, gt_wav)

    print(f"Magnitude Loss: {metrics['mag_loss']}")
    print(f"Envelope Loss: {metrics['env_loss']}")


if __name__ == "__main__":
    pred_wav, sr = librosa.load("path_to_predicted_audio.wav", sr=22050)
    gt_wav, sr = librosa.load("path_to_ground_truth_audio.wav", sr=22050)

    # metric 계산
    evaluate(pred_wav, gt_wav)
