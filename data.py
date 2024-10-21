import os
import math
import json
import random
import pickle
import einops
import librosa
import numpy as np
from tqdm import tqdm
from PIL import Image
import soundfile as sf

import torch
import torchaudio
import torchvision.transforms as T

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import os
import numpy.typing as npt
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio

from dataparser import SoundSpacesDataparserOutputs

import librosa
import scipy.io.wavfile as wavfile

class RWAVSDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 sr=22050,
                 no_pos=False,
                 no_ori=False):
        super(RWAVSDataset, self).__init__()
        self.split = split
        self.sr = sr

        clip_len = 0.5 # second
        wav_len = int(2 * clip_len * sr)

        # sound source
        position = json.loads(open(os.path.join(os.path.dirname(data_root[:-1]), "position.json"), "r").read())
        position = np.array(position[data_root.split('/')[-2]]["source_position"][:2]) # (x, y)
        print(f"Split: {split}, sound source: {position}, wav_len: {wav_len}")

        # rgb and depth features
        feats = pickle.load(open(os.path.join(data_root, f"feats_{split}.pkl"), "rb"))

        # audio
        if os.path.exists(os.path.join(data_root, "binaural_syn_re.wav")):
            audio_bi, _ = librosa.load(os.path.join(data_root, "binaural_syn_re.wav"), sr=sr, mono=False)
        else:
            print("Unavilable, re-process binaural...")
            audio_bi_path = os.path.join(data_root, "binaural_syn.wav")
            audio_bi, _ = librosa.load(audio_bi_path, sr=sr, mono=False) # [2, ?]
            audio_bi = audio_bi / np.abs(audio_bi).max()
            sf.write(os.path.join(data_root, "binaural_syn_re.wav"), audio_bi.T, sr, 'PCM_16')
        
        if os.path.exists(os.path.join(data_root, "source_syn_re.wav")):
            audio_sc, _ = librosa.load(os.path.join(data_root, "source_syn_re.wav"), sr=sr, mono=True)
        else:
            print("Unavilable, re-process source...")
            audio_sc_path = os.path.join(data_root, "source_syn.wav")
            audio_sc, _ = librosa.load(audio_sc_path, sr=sr, mono=True) # [?]
            audio_sc = audio_sc / np.abs(audio_sc).max()
            sf.write(os.path.join(data_root, "source_syn_re.wav"), audio_sc.T, sr, 'PCM_16')

        # pose
        transforms_path = os.path.join(data_root, f"transforms_scale_{split}.json")
        transforms = json.loads(open(transforms_path, "r").read())

        # data
        data_list = []
        for item_idx, item in enumerate(transforms["camera_path"]):
            pose = np.array(item["camera_to_world"]).reshape(4, 4)
            xy = pose[:2,3]
            ori = pose[:2,2]
            data = {"pos": xy}
            ori = relative_angle(position, xy, ori)
            data["ori"] = ori

            if no_pos:
                data["pos"] = np.zeros(2)
            
            if no_ori:
                data["ori"] = 0

            data["rgb"] = feats["rgb"][item_idx]
            data["depth"] = feats["depth"][item_idx]

            # extract key frames at 1 fps
            time = int(item["file_path"].split('/')[-1].split('.')[0])
            data["img_idx"] = time
            st_idx = max(0, int(sr * (time - clip_len)))
            ed_idx = min(audio_bi.shape[1]-1, int(sr * (time + clip_len)))
            if ed_idx - st_idx < int(clip_len * sr): continue
            audio_bi_clip = audio_bi[:, st_idx:ed_idx]
            audio_sc_clip = audio_sc[st_idx:ed_idx]

            # padding with zero
            if(ed_idx - st_idx < wav_len):
                pad_len = wav_len - (ed_idx - st_idx)
                audio_bi_clip = np.concatenate((audio_bi_clip, np.zeros((2, pad_len))), axis=1)
                audio_sc_clip = np.concatenate((audio_sc_clip, np.zeros((pad_len))), axis=0)
                print(f"padding from {ed_idx - st_idx} -> {wav_len}")
            elif(ed_idx - st_idx > wav_len):
                audio_bi_clip = audio_bi_clip[:, :wav_len]
                audio_sc_clip = audio_sc_clip[:wav_len]
                print(f"cutting from {ed_idx - st_idx} -> {wav_len}")

            # binaural
            spec_bi = stft(audio_bi_clip)
            mag_bi = np.abs(spec_bi) # [2, T, F]
            phase_bi = np.angle(spec_bi) # [2, T, F]
            data["mag_bi"] = mag_bi

            # source
            spec_sc = stft(audio_sc_clip)
            mag_sc = np.abs(spec_sc) # [T, F]
            phase_sc = np.angle(spec_sc) # [T, F]
            data["mag_sc"] = mag_sc

            data["wav_bi"] = audio_bi_clip
            data["phase_bi"] = phase_bi
            data["wav_sc"] = audio_sc_clip
            data["phase_sc"] = phase_sc

            data_list.append(data)
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

def vector_angle(xy):
    radians = math.atan2(xy[0], xy[1])
    return radians / (1.01 * np.pi) # trick to make sure ori in open set (-1, 1)

def relative_angle(source, xy, ori): # (-1, 1)
    s = source - xy
    s = s / np.linalg.norm(s)
    d = ori / np.linalg.norm(ori)
    theta = np.arccos(np.clip(np.dot(s, d), -1, 1)) / (1.01 * np.pi)
    rho = np.arcsin(np.clip(np.cross(s, d), -1, 1))
    if rho < 0:
        theta *= -1
    return theta

def stft(signal):
    spec = librosa.stft(signal, n_fft=512)
    if spec.ndim == 2:
        spec = spec.T
    elif spec.ndim == 3:
        spec = einops.rearrange(spec, "c f t -> c t f")
    else:
        raise NotImplementedError
    return spec






""""""""""""" Sound Spaces """""""""""""
class SoundSpacesDataset(Dataset):
    """Dataset that returns audios.

    Args:
        dataparser_outputs: description of where and how to read input audios.
    """

    exclude_batch_keys_from_device: List[str] = ["audios"]

    def __init__(self, dataparser_outputs: SoundSpacesDataparserOutputs, 
                 mode: Literal["train", "eval", "inference"] = "train",
                 max_len: int = 100, 
                 mag_path: Path = None, 
                 wav_path: Path = None,
                 fs: int = 22050,
                 hop_len: int = 128, 
                 mean = None,
                 std = None,):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.audios_filenames=self._dataparser_outputs.audios_filenames
        self.microphone_poses=self._dataparser_outputs.microphone_poses
        self.source_poses=self._dataparser_outputs.source_poses
        self.microphone_rotations=self._dataparser_outputs.microphone_rotations
        self.scene_box=self._dataparser_outputs.scene_box

        self.mag_path = mag_path    
        self.wav_path = wav_path

        self.fs = fs
        self.hop_len = hop_len

        self.max_len = max_len  
        self.max_len_time = self.max_len * self.hop_len 

        self.mode = mode

        self.mean = mean
        self.std = std

    def __len__(self):
        if self.mode == "train":
            return len(self._dataparser_outputs.audios_filenames) * self.max_len
        elif self.mode == "eval":
            return len(self._dataparser_outputs.audios_filenames) * self.max_len
        elif self.mode == 'inference': 
            return len(self._dataparser_outputs.audios_filenames) 
        return len(self._dataparser_outputs.audios_filenames)
    
    def get_id_tmp(self, idx):
        return idx//self.max_len, idx%self.max_len
    
    def get_data(self, audio_idx: int):
        """Returns the of shape STFT slices of size(C, F), microphone pose, source pose, and microphone rotation.
        """
        
        stft_id, stft_tp = self.get_id_tmp(audio_idx)
        audio_filename = self._dataparser_outputs.audios_filenames[stft_id]

        # Load STFT
        stft = np.load(os.path.join(self.mag_path, audio_filename + '.npy'))
        stft = torch.from_numpy(stft)
        if stft_tp < stft.shape[2]:
            stft = torch.log(stft[:,:,stft_tp] + 1e-3)
        else:
            min_value = stft.min()
            stft = torch.ones(stft.shape[0], stft.shape[1]) * min_value
            stft = torch.log(stft + 1e-3)

        # Get poses from dataparser
        microphone_pose = self._dataparser_outputs.microphone_poses[stft_id]
        source_pose = self._dataparser_outputs.source_poses[stft_id]
        microphone_rotation = self._dataparser_outputs.microphone_rotations[stft_id]

        data = {"audio_idx": stft_id, "data": stft, "time_query": stft_tp,
                'rot': microphone_rotation, 'mic_pose': microphone_pose, 'source_pose': source_pose}
        return data


    def get_data_eval(self, audio_idx: int):
        """Returns the STFT of shape (C, F, T), microphone pose, source pose, and microphone rotation.

        Args:
            audio_idx: The audio index in the dataset.
        """
        audio_filename = self._dataparser_outputs.audios_filenames[audio_idx]

        if self.mode == "inference":
            # all zeros stft because we don't have GT
            stft = torch.zeros((2, 257, self.max_len))
            waveform = torch.zeros((2, int(self.max_len_time)))
        
        else:
            # Load STFT
            stft = np.load(os.path.join(self.mag_path, audio_filename + '.npy'))
            if stft.shape[2] > self.max_len:
                stft = stft[:,:,:self.max_len]
            else:
                min_value = stft.min()
                stft = np.pad(stft, ((0,0), (0,0), (0, self.max_len - stft.shape[2])), 'constant', constant_values=min_value)

            stft = np.log(stft + 1e-3)
            stft = torch.from_numpy(stft) 
            
            # Load GT waveform for metric computation
            loaded = wavfile.read(os.path.join(self.wav_path, audio_filename + '.wav'))
            waveform = np.clip(loaded[1], -1.0, 1.0).T

            if waveform.shape[1] == 0:
                waveform = np.zeros((2, int(self.fs*0.5)))

            if self.fs != 44100:
                init_fs = 44100
                if waveform.shape[1]<int(init_fs*0.1):
                    padded_wav = librosa.util.fix_length(data=waveform, size=int(init_fs*0.1))
                    resampled_wav= librosa.resample(padded_wav, orig_sr=init_fs, target_sr=self.fs)
                else:
                    resampled_wav= librosa.resample(waveform, orig_sr=init_fs, target_sr=self.fs)
                waveform = resampled_wav

            if waveform.shape[1] > self.max_len_time:
                waveform = waveform[:,:int(self.max_len_time)]
            else:
                waveform = np.pad(waveform, ((0,0), (0, self.max_len_time - waveform.shape[1])), 'constant', constant_values=0)

            waveform = torch.from_numpy(waveform)

        # Get poses from dataparser
        microphone_pose = self._dataparser_outputs.microphone_poses[audio_idx]
        source_pose = self._dataparser_outputs.source_poses[audio_idx]
        microphone_rotation = self._dataparser_outputs.microphone_rotations[audio_idx]

        data = {"audio_idx": audio_idx, "data": stft, 'waveform': waveform, 
                'rot': microphone_rotation, 'mic_pose': microphone_pose, 'source_pose': source_pose}

        return data

    def __getitem__(self, image_idx: int) -> Dict:
        if self.mode == "train":
            return self.get_data(image_idx)
        elif self.mode == "eval":
            return self.get_data(image_idx) # special case for evaluation on STFT slices
        return self.get_data_eval(image_idx)

    @property
    def audio_filenames(self) -> List[Path]:
        """
        Returns audio filenames for this dataset.
        """
        return self._dataparser_outputs.audio_filenames