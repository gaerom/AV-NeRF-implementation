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

        
        """
        ### llava text embedding 
        text_embedding_file = os.path.join(data_root, 'llava_responses', f"embedding_{split}.pkl")
        with open(text_embedding_file, "rb") as f:
            self.text_embeddings = pickle.load(f)
        """
        

        clip_len = 0.5 # audio clip length, 0.5 second
        wav_len = int(2 * clip_len * sr) # 22050 samples

        # sound source
        position = json.loads(open(os.path.join(os.path.dirname(data_root[:-1]), "position.json"), "r").read())
        position = np.array(position[data_root.split('/')[-2]]["source_position"][:2]) # (x, y)
        ### "1": {"source_position": [0.06, 0.13, -0.45]}
        print(f"Split: {split}, sound source: {position}, wav_len: {wav_len}")

        # rgb and depth features
        feats = pickle.load(open(os.path.join(data_root, f"feats_{split}.pkl"), "rb"))

        ### rgb(GT) + depth (upper bound 확인용)
        # feats = pickle.load(open(os.path.join(data_root + "GT", f"feats_GT_{split}.pkl"), "rb"))

        
        
        # audio
        if os.path.exists(os.path.join(data_root, "binaural_syn_re.wav")):
            audio_bi, _ = librosa.load(os.path.join(data_root, "binaural_syn_re.wav"), sr=sr, mono=False)
        else: # 없으면 resampling
            print("Unavilable, re-process binaural...")
            audio_bi_path = os.path.join(data_root, "binaural_syn.wav")
            audio_bi, _ = librosa.load(audio_bi_path, sr=sr, mono=False) # [2, ?]
            audio_bi = audio_bi / np.abs(audio_bi).max()
            sf.write(os.path.join(data_root, "binaural_syn_re.wav"), audio_bi.T, sr, 'PCM_16')
        
        
        # source audio
        if os.path.exists(os.path.join(data_root, "source_syn_re.wav")):
            audio_sc, _ = librosa.load(os.path.join(data_root, "source_syn_re.wav"), sr=sr, mono=True)
        else:
            print("Unavilable, re-process source...")
            audio_sc_path = os.path.join(data_root, "source_syn.wav")
            audio_sc, _ = librosa.load(audio_sc_path, sr=sr, mono=True) # [?]
            audio_sc = audio_sc / np.abs(audio_sc).max()
            sf.write(os.path.join(data_root, "source_syn_re.wav"), audio_sc.T, sr, 'PCM_16')

        # pose (listener, camera pose)
        transforms_path = os.path.join(data_root, f"transforms_scale_{split}.json")
        transforms = json.loads(open(transforms_path, "r").read())


        # data
        data_list = []
        for item_idx, item in enumerate(transforms["camera_path"]): # item_idx: transforms["camera_path"]에서 현재 처리중인 idx
            pose = np.array(item["camera_to_world"]).reshape(4, 4)
            """
            [[-0.7845,  0.0204,  0.6198, -0.1690],
            [ 0.6200,  0.0069,  0.7846, -0.4206],
            [ 0.0117,  0.9998, -0.0181,  0.0402],
            [ 0.0,     0.0,     0.0,     1.0]]
            
            """
            xy = pose[:2,3] # [-0.1690, -0.4206], camera pose (x, y)
            ori = pose[:2,2] # [0.6198, 0.7846]
            data = {"pos": xy}
            # source position, camera 위치, camera viewing direction
            # relative angle 계산
            ori = relative_angle(position, xy, ori)
            data["ori"] = ori

            if no_pos:
                data["pos"] = np.zeros(2)
            
            if no_ori:
                data["ori"] = 0

            data["rgb"] = feats["rgb"][item_idx] # item_idx: feats["rgb"]에서 해당하는 index를 가져와서 -> data['rgb']
            data["depth"] = feats["depth"][item_idx]
            
            """
            ### llava embedding 추가 ### 
            image_name = item["file_path"].split('/')[-1]
            text_embedding = self.text_embeddings.get(image_name)

            data["text_embedding"] = text_embedding # [1024] -> concat 하기 전에 [B, 1024]로 바꿔야 함. 
            """
            


            # extract key frames at 1 fps
            time = int(item["file_path"].split('/')[-1].split('.')[0]) # 00025 -> 25
            data["img_idx"] = time # 25
            # start index
            st_idx = max(0, int(sr * (time - clip_len))) # time 기준으로 0.5s 앞
            # end index
            ed_idx = min(audio_bi.shape[1]-1, int(sr * (time + clip_len)))
            if ed_idx - st_idx < int(clip_len * sr): continue # 유효하지 않은 구간 pass
            audio_bi_clip = audio_bi[:, st_idx:ed_idx] # 2-channels
            audio_sc_clip = audio_sc[st_idx:ed_idx] # 1-channel

            # padding with zero -> audio clip 길이 맞추기
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
            spec_bi = stft(audio_bi_clip) # (2, 173, 257)
            mag_bi = np.abs(spec_bi) # [2, T, F], magnitude [2, time, frequency]
            phase_bi = np.angle(spec_bi) # [2, T, F], phase
            data["mag_bi"] = mag_bi ### GT binaural magnitude

            # source
            spec_sc = stft(audio_sc_clip) # (173, 257)
            mag_sc = np.abs(spec_sc) # [T, F], 1-channel
            phase_sc = np.angle(spec_sc) # [T, F]
            data["mag_sc"] = mag_sc # 나중에 mask가 결합될 부분(?_

            data["wav_bi"] = audio_bi_clip
            data["phase_bi"] = phase_bi
            data["wav_sc"] = audio_sc_clip
            data["phase_sc"] = phase_sc

            data_list.append(data) 
            # "pos", "ori", "rgb", "depth", "img_idx
            # "mag_bi", "wav_bi", "phase_bi" / "mag_sc", "wav_sc", "phase_sc"
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
