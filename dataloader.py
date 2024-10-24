import os
import json
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

### DataLoader for SoundSpaces (수정중)

# class SoundSpaces:
#     def __init__(self, dataset_dir, split_file):
#         # dataset_dir: 데이터셋이 저장된 디렉토리 경로
#         # split_file: train/test 정보가 저장된 split.json 파일 경로
#         self.dataset_dir = dataset_dir
#         self.split_file = split_file
#         self.load_split()

#     def load_split(self):
#         # split.json 파일에서 train/test 파일 목록 로드
#         with open(self.split_file, 'r') as f:
#             self.split_data = json.load(f)

#     def load_data(self, split='train'):
#         files = []  # 학습 또는 테스트에 사용될 파일 리스트
#         mic_poses = []
#         source_poses = []
#         rots = []

#         # split.json 파일에서 train/test에 해당하는 데이터를 가져옴
#         split_list = self.split_data[split]

#         for f in split_list:
#             # f: "180/143_90" -> rotation과 source/mic 위치 정보를 추출
#             rot, r_s = f.split('/')  # 예: "180/143_90" -> rot=180, r_s=143_90
#             rot = int(rot)  # 회전 정보는 각도
#             mic_id, source_id = r_s.split('_')  # 마이크 위치 ID와 소스 위치 ID
#             mic_pose = np.array([float(mic_id), 0, 0])  # 마이크 위치
#             source_pose = np.array([float(source_id), 0, 0])  # 소스 위치

#             mic_pose = np.expand_dims(mic_pose, axis=0)
#             source_pose = np.expand_dims(source_pose, axis=0)
            
#             # 데이터셋 경로에서 .npy 파일 이름 생성
#             npy_file = f"{mic_id}_{source_id}.npy"
#             npy_path = os.path.join(self.dataset_dir, str(rot), npy_file)

#             # 각도를 radian으로 변환 후 방향 벡터 계산
#             rot = np.deg2rad(float(rot))
#             rot = np.array([np.cos(rot), 0, np.sin(rot)])  # 방향 코사인 계산
#             rot = (rot + 1.0) / 2.0  # 0~1로 정규화
#             rot = np.expand_dims(rot, axis=0)


#             # 파일 경로와 관련된 정보들을 리스트에 저장
#             mic_poses.append(mic_pose)
#             source_poses.append(source_pose)
#             rots.append(rot)
#             files.append(npy_path)  # 해당 파일 경로 저장

#         # 최종 데이터를 배열로 변환
#         mic_poses = np.concatenate(mic_poses, axis=0)
#         source_poses = np.concatenate(source_poses, axis=0)
#         rots = np.concatenate(rots, axis=0)

#         return {
#             'rot': rots,  # 방향 정보
#             'mic_pose': mic_poses,  # 마이크 위치
#             'source_pose': source_poses,  # 소스 위치
#             'files': files  # 로드된 GT 파일 경로들
#         }

#     def load_gt(self, file_path):
#         # .npy 파일에서 GT 로드
#         return np.load(file_path)


# dataset_dir = 'SoundSpaces/apartment_1/binaural_magnitudes_sr22050'
# split_file = 'SoundSpaces/apartment_1/metadata_AudioNeRF/split.json' # scene -> 나중에 args로 설정
# data_loader = SoundSpaces(dataset_dir, split_file)

# # Train 데이터 로드
# train_data = data_loader.load_data(split='train') # train / test

# def load_gt_parallel(file_path):
#     return data_loader.load_gt(file_path)

# # Ground Truth 데이터 로드
# # for file_path in tqdm(train_data['files'], desc="Loading Ground Truth"):
# #     gt_data = data_loader.load_gt(file_path)

# with ThreadPoolExecutor(max_workers=10) as executor:
#     gt_data_list = list(tqdm(executor.map(load_gt_parallel, train_data['files']), total=len(train_data['files']), desc="Loading Ground Truth"))



class SoundSpaces(Dataset):
    def __init__(self, dataset_dir, split_file, device='cuda'):
        # dataset_dir: 데이터셋 디렉토리 경로
        # split_file: train/test split 파일 경로
        self.dataset_dir = dataset_dir
        self.split_file = split_file
        self.device = device
        self.load_split()
        
        clip_len = 0.5 # second
        wav_len = int(2 * clip_len * 22050)

    def load_split(self):
        
        with open(self.split_file, 'r') as f:
            self.split_data = json.load(f)

    def load_data(self, split='train'):
        self.files = []
        self.mic_poses = []
        self.source_poses = []
        self.rots = []

        split_list = self.split_data[split]

        for f in split_list:
            rot, r_s = f.split('/')
            rot = int(rot)
            mic_id, source_id = r_s.split('_')
            mic_pose = np.array([float(mic_id), 0, 0])
            source_pose = np.array([float(source_id), 0, 0])
            mic_pose = np.expand_dims(mic_pose, axis=0)
            source_pose = np.expand_dims(source_pose, axis=0)

            npy_file = f"{mic_id}_{source_id}.npy"
            npy_path = os.path.join(self.dataset_dir, str(rot), npy_file)

            # 각도를 radian으로 변환한 후 방향 코사인 계산
            rot = np.deg2rad(float(rot))
            rot = np.array([np.cos(rot), 0, np.sin(rot)])
            rot = (rot + 1.0) / 2.0  
            rot = np.expand_dims(rot, axis=0)

            self.mic_poses.append(mic_pose)
            self.source_poses.append(source_pose)
            self.rots.append(rot)
            self.files.append(npy_path)

        self.mic_poses = np.concatenate(self.mic_poses, axis=0)
        self.source_poses = np.concatenate(self.source_poses, axis=0)
        self.rots = np.concatenate(self.rots, axis=0)
        
        return self

    def load_gt(self, file_path):
        data = np.load(file_path)
        return torch.tensor(data, device=self.device, dtype=torch.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        gt_data = self.load_gt(file_path)
        rot = torch.tensor(self.rots[idx], device=self.device)
        mic_pose = torch.tensor(self.mic_poses[idx], device=self.device)
        source_pose = torch.tensor(self.source_poses[idx], device=self.device)

        return {
            'rot': rot,
            'mic_pose': mic_pose,
            'source_pose': source_pose,
            'gt_data': gt_data
        }


def pad_collate_fn(batch):
    max_len = max([d['gt_data'].shape[-1] for d in batch])  # batch: 16 - max_len: 62

    for i in range(len(batch)):
        gt_data = batch[i]['gt_data'] # [2, 257, ?], 16 - 56, 53, 59...
        if gt_data.shape[-1] < max_len: 
            pad_size = max_len - gt_data.shape[-1]
            batch[i]['gt_data'] = torch.nn.functional.pad(gt_data, (0, pad_size)) 

    return torch.utils.data.dataloader.default_collate(batch)





dataset_dir = 'SoundSpaces/apartment_test/binaural_magnitudes_sr22050'
split_file = 'SoundSpaces/apartment_test/metadata_AudioNeRF/split.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


dataset = SoundSpaces(dataset_dir, split_file, device=device)
dataset.load_data(split='train')  # train / test 데이터 로드


data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=pad_collate_fn)


for batch in tqdm(data_loader, desc="Loading data onto GPU"):
    # 각 batch에서 'rot', 'mic_pose', 'source_pose', 'gt_data'를 가져옴
    rot = batch['rot']
    mic_pose = batch['mic_pose']
    source_pose = batch['source_pose']
    gt_data = batch['gt_data']

print()