import os
import sys
import math
import pickle
import einops
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.distributed as dist

import soundfile as sf


### ver.1
# from utils import energy_decay, istft, Evaluator

# class Trainer(object):
#     def __init__(self,
#                  args,
#                  model,
#                  criterion,
#                  optimizer,
#                  log_dir,
#                  last_epoch=-1,
#                  last_iter=-1,
#                  device='cuda',
#                 ):
#         self.args = args
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         if log_dir:
#             self.log_dir = os.path.join(log_dir, self.args.output_dir if self.args.output_dir else f"{self.args.room_name}_{self.args.batch_size}_{self.args.lr}/")
#         self.epoch = last_epoch + 1
#         self.max_epoch = self.args.max_epoch
#         self.device = device
#         self.iter_count = last_iter + 1
#         if self.optimizer is not None:
#             self.writer = SummaryWriter(self.log_dir)

#     def train(self, train_loader):
#         self.model.train()
#         t = tqdm(total=len(train_loader), desc=f"[EPOCH {self.epoch} TRAIN]", leave=False)
#         self.writer.add_scalar("epoch", self.epoch, self.epoch)
#         for data in train_loader:
#             for k in data.keys():
#                data[k] = data[k].float().to(self.device)
#             ret = self.model(data)

#             # mag (A-NeRF loss)
#             mag_bi_mean = data["mag_bi"].mean(1)
#             loss_mono = F.mse_loss(ret["reconstr_mono"], mag_bi_mean)
#             loss_bi = F.mse_loss(ret["reconstr"], data["mag_bi"]) # s_l, s_r 관련 loss
#             loss = loss_mono + loss_bi # final loss

#             if self.args.wave:
#                 env_prd = myhibert(ret["wav"].flatten(0, 1)).abs()
#                 env_gt = myhibert(data["wav_bi"].flatten(0, 1)).abs()
#                 loss_wave = torch.sqrt(torch.mean(torch.pow(env_prd - env_gt, 2)) + 1e-7)
#                 loss = loss + loss_wave
#                 print(f"Use wave loss: {loss_mono} {loss_bi} {loss_wave} {loss}")

#             # energy
#             if self.args.energy:
#                 loss_eng_mono = 0.01 * energy_decay(ret["reconstr_mono"], mag_bi_mean)
#                 loss_eng_bi = 0.01 * energy_decay(ret["reconstr"].flatten(0, 1), data["mag_bi"].flatten(0, 1))
#                 loss = loss + loss_eng_mono + loss_eng_bi

#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             # adjust lr
#             warmup_step = int(0.1 * self.args.max_epoch) * len(train_loader)
#             if self.iter_count < warmup_step:
#                 lr = self.args.lr * self.iter_count / warmup_step
#             else:
#                 lr = self.args.lr * 0.1 ** (2 * (self.iter_count - warmup_step) / (self.args.max_epoch * len(train_loader) - warmup_step))
#             self.optimizer.param_groups[0]["lr"] = lr

#             self.writer.add_scalar("train/lr", lr, self.iter_count)
#             self.writer.add_scalar("train/loss_mono", loss_mono, self.iter_count)
#             self.writer.add_scalar("train/loss_bi", loss_bi, self.iter_count)
#             if self.args.energy:
#                 self.writer.add_scalar("train/loss_eng_mono", loss_eng_mono, self.iter_count)
#                 self.writer.add_scalar("train/loss_eng_bi", loss_eng_bi, self.iter_count)
#             t.update()
#             self.iter_count += 1
#         t.close()

#     def eval(self, val_loader, save=False):
#         self.model.eval()
#         evaluator = Evaluator()
#         save_list = []
#         with torch.no_grad():
#             t = tqdm(total=len(val_loader), desc=f"[EPOCH {self.epoch} EVAL]", leave=False)
#             for data_idx, data in enumerate(val_loader):
#                 for k in data.keys():
#                     data[k] = data[k].float().to(self.device)
#                 ret = self.model(data)

#                 for b in range(data["mag_bi"].shape[0]):
#                     mag_prd = ret["reconstr"][b].cpu().numpy()
#                     phase_prd = data["phase_sc"][b].cpu().numpy()
#                     spec_prd = mag_prd * np.exp(1j * phase_prd[np.newaxis,:])
#                     wav_prd = librosa.istft(spec_prd.transpose(0, 2, 1), length=22050)
#                     mag_gt = data["mag_bi"][b].cpu().numpy()
#                     wav_gt = data["wav_bi"][b].cpu().numpy()

#                     # # save predicted binaural audios
#                     # save_path = os.path.join(self.args.result_dir, "blender", f"binaural_audio_{data_idx}_{b}.wav")
#                     # sf.write(save_path, wav_prd.T, 22050)

#                     # # save gt audios
#                     # save_gt_path = os.path.join(self.args.result_dir, "blender", f"binaural_audio_gt_{data_idx}_{b}.wav")
#                     # sf.write(save_gt_path, wav_gt.T, 22050)

#                     loss_list = evaluator.update(mag_prd, mag_gt, wav_prd, wav_gt)
#                     if save:
#                         save_list.append({"wav_prd": wav_prd,
#                                           "wav_gt": wav_gt,
#                                           "loss": loss_list,
#                                           "img_idx": data["img_idx"][b].cpu().numpy()})
#                 t.update()
#             t.close()
#         result = evaluator.report()
#         if hasattr(self, "writer"):
#             for k, v in result.items():
#                 self.writer.add_scalar(f"eval/{k}", v, self.epoch)
        
#         if save:
#             return result, save_list
#         else:
#             return result

#     def save_ckpt(self):
#         try:
#             state_dict = self.model.module.state_dict()  # remove prefix of multi GPUs
#         except AttributeError:
#             state_dict = self.model.state_dict()
        
#         if not os.path.exists(self.log_dir):
#             os.mkdir(self.log_dir)
#         torch.save({
#                 'epoch': self.epoch,
#                 'iter': self.iter_count,
#                 'state_dict': state_dict,
#                 'optimizer': self.optimizer.state_dict()},
#                 os.path.join(self.log_dir, f"{self.epoch}.pth"))

################################### binaural audio (RWAVS) ##########################################

import torch
import torch.nn.functional as F
import torchaudio

from utils import SoundSpacesEvaluator

# class Trainer(object):
#     def __init__(self,
#                  args,
#                  model,
#                  criterion,
#                  optimizer,
#                  log_dir,
#                  last_epoch=-1,
#                  last_iter=-1,
#                  device='cuda',
#                 ):
#         self.args = args
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         if log_dir:
#             self.log_dir = os.path.join(log_dir, self.args.output_dir if self.args.output_dir else f"{self.args.room_name}_{self.args.batch_size}_{self.args.lr}/")
#         self.epoch = last_epoch + 1
#         self.max_epoch = self.args.max_epoch
#         self.device = device
#         self.iter_count = last_iter + 1
#         if self.optimizer is not None:
#             self.writer = SummaryWriter(self.log_dir)

#         # STFT 파라미터 설정
#         self.n_fft = 512
#         self.win_length = 512
#         self.hop_length = 128
#         self.stft_transform = torch.stft  # STFT 변환
#         self.griffin_lim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        
#         self.evaluator = SoundSpacesEvaluator(fs=22050)

#     def train(self, train_loader):
#         self.model.train()
#         t = tqdm(total=len(train_loader), desc=f"[EPOCH {self.epoch} TRAIN]", leave=False)
#         self.writer.add_scalar("epoch", self.epoch, self.epoch)
#         for data in train_loader:
#             for k in data.keys():
#                data[k] = data[k].float().to(self.device)
#             ret = self.model(data)

#             # Ground truth와 예측값을 STFT로 변환
#             mag_bi_gt = torch.stft(data["wav_bi"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
#             mag_bi_pred = torch.stft(ret["reconstr"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
            
#             # 크기 정보만 사용해 L2 Loss 계산
#             gt_magnitude = torch.abs(mag_bi_gt)
#             pred_magnitude = torch.abs(mag_bi_pred)
#             loss = F.mse_loss(pred_magnitude, gt_magnitude)  # L2 Loss

#             # Griffin-Lim으로 시간 도메인 신호 복원 (저장 및 분석용)
#             wav_pred_left = self.griffin_lim(ret["left"])
#             wav_pred_right = self.griffin_lim(ret["right"])
#             pred_rir = torch.stack([wav_pred_left, wav_pred_right], dim=1)  # [B, 2, T]

#             # 최종 Loss 업데이트 및 최적화
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             # 학습 기록 저장
#             self.writer.add_scalar("train/loss", loss, self.iter_count)
#             t.update()
#             self.iter_count += 1
#         t.close()

#     def eval(self, val_loader, save=False):
#         self.model.eval()
        
#         with torch.no_grad():
#             t = tqdm(total=len(val_loader), desc=f"[EPOCH {self.epoch} EVAL]", leave=False)
#             for data_idx, data in enumerate(val_loader):
#                 for k in data.keys():
#                     data[k] = data[k].float().to(self.device)
#                 ret = self.model(data)

#                 # Ground truth와 예측값을 STFT로 변환
#                 mag_bi_gt = torch.stft(data["wav_bi"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
#                 mag_bi_pred = torch.stft(ret["reconstr"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)

#                 # 크기 정보만 사용해 L2 Loss 계산
#                 gt_magnitude = torch.abs(mag_bi_gt)
#                 pred_magnitude = torch.abs(mag_bi_pred)
#                 loss = F.mse_loss(pred_magnitude, gt_magnitude)  # L2 Loss

#                 # Griffin-Lim으로 시간 도메인 신호 복원
#                 wav_pred_left = self.griffin_lim(ret["left"])
#                 wav_pred_right = self.griffin_lim(ret["right"])
#                 pred_rir = torch.stack([wav_pred_left, wav_pred_right], dim=1)  # [B, 2, T]

#                 # 평가 결과 저장 및 리포트
#                 self.evaluator.update(pred_magnitude.cpu().numpy(), gt_magnitude.cpu().numpy(), pred_rir.cpu().numpy(), data["wav_bi"].cpu().numpy())
#                 t.update()
#             t.close()
#         result = self.evaluator.report()
#         if hasattr(self, "writer"):
#             for k, v in result.items():
#                 self.writer.add_scalar(f"eval/{k}", v, self.epoch)
#         return result

#     def save_ckpt(self):
#         try:
#             state_dict = self.model.module.state_dict()  # remove prefix of multi GPUs
#         except AttributeError:
#             state_dict = self.model.state_dict()
        
#         if not os.path.exists(self.log_dir):
#             os.mkdir(self.log_dir)
#         torch.save({
#                 'epoch': self.epoch,
#                 'iter': self.iter_count,
#                 'state_dict': state_dict,
#                 'optimizer': self.optimizer.state_dict()},
#                 os.path.join(self.log_dir, f"{self.epoch}.pth"))


import torch
import torch.nn.functional as F
import torchaudio
from utils import SoundSpacesEvaluator

class Trainer(object):
    def __init__(self,
                 args,
                 model,
                 criterion,
                 optimizer,
                 log_dir,
                 last_epoch=-1,
                 last_iter=-1,
                 device='cuda'):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if log_dir:
            self.log_dir = os.path.join(log_dir, self.args.output_dir if self.args.output_dir else f"{self.args.batch_size}_{self.args.lr}/")

        self.epoch = last_epoch + 1
        self.max_epoch = self.args.max_epoch
        self.device = device
        self.iter_count = last_iter + 1
        if self.optimizer is not None:
            self.writer = SummaryWriter(self.log_dir)

        # STFT 파라미터 설정
        self.n_fft = 512
        self.win_length = 512
        self.hop_length = 128
        self.stft_transform = torch.stft  # STFT 변환
        self.griffin_lim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        
        self.evaluator = SoundSpacesEvaluator(fs=22050)

    def train(self, train_loader):
        self.model.train()
        t = tqdm(total=len(train_loader), desc=f"[EPOCH {self.epoch} TRAIN]", leave=False)
        self.writer.add_scalar("epoch", self.epoch, self.epoch)
        
        for data in train_loader:
            for k in data.keys():
                data[k] = data[k].float().to(self.device)

            # A-NeRF 모델의 예측 값 가져오기
            ret = self.model(data)

            # Ground truth와 예측 값에서 크기 정보 추출
            mag_bi_gt = torch.abs(data["gt_data"])  # [B, 2, 257, 69] 형태로 추출
            
            # 예측 신호에 대해 STFT 적용
            mag_bi_pred_left = torch.stft(ret["left"], n_fft=self.n_fft, hop_length=self.hop_length, 
                                        win_length=self.win_length, return_complex=True)  # 복소수 형태로 STFT 수행
            mag_bi_pred_right = torch.stft(ret["right"], n_fft=self.n_fft, hop_length=self.hop_length, 
                                        win_length=self.win_length, return_complex=True)  # 복소수 형태로 STFT 수행
            
            # STFT 결과를 복소수 텐서에서 실수 텐서로 변환
            mag_bi_pred_left = torch.view_as_real(mag_bi_pred_left)  # [B, 257, ?, 2]
            mag_bi_pred_right = torch.view_as_real(mag_bi_pred_right)  # [B, 257, ?, 2]

            # 예측된 스펙트로그램 결합
            mag_bi_pred = torch.cat([mag_bi_pred_left.unsqueeze(1), 
                                    mag_bi_pred_right.unsqueeze(1)], dim=1)  # [B, 2, 257, ?, 2]

            # 예측 값에서 마지막 차원을 제거하여 크기를 맞춤
            pred_magnitude = mag_bi_pred[..., 0]  # [B, 2, 257, ?]
            
            # Ground truth 크기 정보 추출
            gt_magnitude = torch.abs(mag_bi_gt)  # [B, 2, 257, 69]

            # Ground truth와 예측 값의 시간 길이를 맞춤
            min_time_len = min(gt_magnitude.shape[-1], pred_magnitude.shape[-1])
            gt_magnitude = gt_magnitude[..., :min_time_len]
            pred_magnitude = pred_magnitude[..., :min_time_len]

            # L2 손실 계산
            loss = F.mse_loss(pred_magnitude, gt_magnitude)

            # 최종 Loss 업데이트 및 최적화
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 학습 기록 저장
            self.writer.add_scalar("train/loss", loss, self.iter_count)
            t.update()
            self.iter_count += 1
        t.close()




    # def eval(self, val_loader, save=False):
    #     self.model.eval()
        
    #     with torch.no_grad():
    #         t = tqdm(total=len(val_loader), desc=f"[EPOCH {self.epoch} EVAL]", leave=False)
    #         for data_idx, data in enumerate(val_loader):
    #             for k in data.keys():
    #                 data[k] = data[k].float().to(self.device)
    #             ret = self.model(data)

    #             # Ground truth와 예측값을 STFT로 변환
    #             # data["gt_data"].shape: [1, 2, 257, 59]

    #             ### add
    #             gt_data = data["gt_data"]
    #             gt_data_for_stft = gt_data[0, :, :, :] # [2, 257, 59]

    #             stft_outputs = []

    #             for i in range(gt_data_for_stft.shape[0]):  # 채널 수만큼 반복
    #                 signal = gt_data_for_stft[i, :, :]  # [257, 59]

    #                 padding = self.n_fft - signal.shape[-1]  ### padding 적용 수정 필요
    #                 if padding > 0:
    #                     signal = torch.nn.functional.pad(signal, (0, padding)) # [257, 512]
                    
    #                 # torch.stft 적용 (입력 데이터가 1D가 아니므로 적절히 수정 필요)
    #                 stft_output = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
    #                 stft_outputs.append(stft_output)
                    
    #                 # print(stft_output.shape)

    #             stft_outputs = torch.stack(stft_outputs, dim=0)

    #             # mag_bi_gt = torch.stft(data["gt_data"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
    #             mag_bi_gt = stft_outputs # [2, 257, 257, 5, 2]
    #             # mag_bi_pred = torch.stft(ret["reconstr"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
                
    #             ################################# pred ##########################################
    #             # ret[left]: [1, 22050]
    #             # ret[right]: [1, 22050]
    #             # for i in range(ret["reconstr"].shape[1]):  # 각 채널에 대해 동일한 과정 적용
    #             #     signal_pred = ret["reconstr"][0, i, :, :]  # [257, t]
    #             #     padding_pred = max(0, self.n_fft - signal_pred.shape[-1])
    #             #     if padding_pred > 0:
    #             #         signal_pred = torch.nn.functional.pad(signal_pred, (0, padding_pred))
                    
    #             #     stft_output_pred = torch.stft(signal_pred, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
    #             #     mag_bi_pred.append(stft_output_pred)

    #             # # stft 결과 결합
    #             # mag_bi_pred = torch.stack(mag_bi_pred, dim=0)

    #             ### ret.keys(): 'left', 'right'
    #             mag_bi_pred_left = torch.stft(ret["left"], n_fft=self.n_fft, hop_length=self.hop_length, 
    #                                           win_length=self.win_length, return_complex=False) # [1, 257, 173, 2]
    #             mag_bi_pred_right = torch.stft(ret["right"], n_fft=self.n_fft, hop_length=self.hop_length, 
    #                                            win_length=self.win_length, return_complex=False) # [1, 257, 173, 2]

    #             mag_bi_pred = torch.cat([mag_bi_pred_left.unsqueeze(1), 
    #                                      mag_bi_pred_right.unsqueeze(1)], dim=1) # [1, 2, 257, 173, 2] -> 2-channels

            
    #             # magnitude L2 Loss
    #             # mag_bi_gt: [2, 257, 257, 5, 2]
    #             # mag_bi_pred: [1, 2, 257, 173, 2]
    #             gt_magnitude = torch.abs(mag_bi_gt)
    #             pred_magnitude = torch.abs(mag_bi_pred)

    #             ####################################### 수정해야 함 #####################################################
    #             min_size_batch = min(gt_magnitude.shape[0], pred_magnitude.shape[0])
    #             min_size_channel = min(gt_magnitude.shape[1], pred_magnitude.shape[1])  
    #             min_size_freq = min(gt_magnitude.shape[-3], pred_magnitude.shape[-3])  

                
    #             gt_magnitude = gt_magnitude[:min_size_batch, :min_size_channel, :min_size_freq, :, :]
    #             pred_magnitude = pred_magnitude[:min_size_batch, :min_size_channel, :min_size_freq, :, :]

    #             min_size_time = min(gt_magnitude.shape[-2], pred_magnitude.shape[-2])
    #             gt_magnitude = gt_magnitude[..., :min_size_time, :]
    #             pred_magnitude = pred_magnitude[..., :min_size_time, :]

    #             ######################################################################################################

    #             loss = F.mse_loss(pred_magnitude, gt_magnitude)  # L2 Loss (차원 일치하도록 수정)

    #             # Griffin-Lim -> 이거 필요 X
    #             # wav_pred_left = self.griffin_lim(ret["left"]) # [1, 22050]
    #             # wav_pred_right = self.griffin_lim(ret["right"]) # [1, 22050]
    #             # pred_rir = torch.stack([wav_pred_left, wav_pred_right], dim=1)  # [B, 2, T]

    #             prd_left_wav = ret["left"]  # left chennal 
    #             prd_right_wav = ret["right"]  # right chennal

    #             prd_stereo_wav = torch.stack([prd_left_wav, prd_right_wav], dim=1)
    #             gt_stereo_wav = data["gt_data"]

    #             # T60, EDT, C50 
    #             metrics = self.evaluator.get_full_metrics(
    #                 prd_stereo_wav.cpu().numpy(),  # 예측 스테레오 오디오 신호
    #                 gt_stereo_wav.cpu().numpy(),   # Ground truth 스테레오 신호
    #                 data["gt_data"].cpu().numpy(),  # Ground truth IR
    #                 prd_stereo_wav.cpu().numpy(),  # 예측 IR
    #                 log_prd=None, log_gt=None    
    #             )

    #             # 메트릭 결과 출력
    #             print(f"T60 Mean Error: {metrics['audio_T60_mean_error']}, EDT: {metrics['audio_EDT']}, C50: {metrics['audio_C50']}")

    #             # 평가 결과 저장 및 리포트
    #             t.update()
    #         t.close()
    #     result = self.evaluator.report()
    #     if hasattr(self, "writer"):
    #         for k, v in result.items():
    #             self.writer.add_scalar(f"eval/{k}", v, self.epoch)
    #     return result
    # def eval(self, val_loader, save=False):
    #     self.model.eval()

    #     with torch.no_grad():
    #         t = tqdm(total=len(val_loader), desc=f"[EPOCH {self.epoch} EVAL]", leave=False)
    #         for data_idx, data in enumerate(val_loader):
    #             for k in data.keys():
    #                 data[k] = data[k].float().to(self.device)

    #             # A-NeRF 모델의 예측 값 가져오기
    #             ret = self.model(data)

    #             # Left, Right 채널로부터 스테레오 오디오 신호 결합
    #             prd_left_wav = ret["left"]  # 왼쪽 채널 (예측 신호)
    #             prd_right_wav = ret["right"]  # 오른쪽 채널 (예측 신호)
    #             prd_stereo_wav = torch.stack([prd_left_wav, prd_right_wav], dim=1)  # 스테레오 신호 결합 [B, 2, T]

    #             # Ground Truth 스테레오 신호 가져오기
    #             gt_stereo_wav = data["gt_data"]  # Ground truth 스테레오 신호 [B, 2, T]

    #             # 메트릭 계산 (T60, EDT, C50)
    #             metrics = self.evaluator.get_full_metrics(
    #                 prd_stereo_wav.cpu().numpy(),  # 예측 스테레오 오디오 신호
    #                 gt_stereo_wav.cpu().numpy(),   # Ground truth 스테레오 신호
    #                 gt_stereo_wav.cpu().numpy(),   # Ground truth IR
    #                 prd_stereo_wav.cpu().numpy(),  # 예측 IR
    #                 gt_stereo_wav.cpu().numpy(),   # Ground truth IR
    #                 log_prd=None, log_gt=None      # 로그 값은 사용하지 않음
    #             )

    #             # 메트릭 결과 출력
    #             print(f"T60 Mean Error: {metrics['audio_T60_mean_error']}, EDT: {metrics['audio_EDT']}, C50: {metrics['audio_C50']}")

    #             # 평가 결과 저장 및 리포트
    #             t.update()
    #         t.close()

    #     # result = self.evaluator.report()
    #     # if hasattr(self, "writer"):
    #     #     for k, v in result.items():
    #     #         self.writer.add_scalar(f"eval/{k}", v, self.epoch)
    #     return

    def eval(self, val_loader, save=False):
        self.model.eval()

        t60_errors = []
        edt_values = []
        c50_values = []

        with torch.no_grad():
            t = tqdm(total=len(val_loader), desc=f"[EPOCH {self.epoch} EVAL]", leave=False)
            for data_idx, data in enumerate(val_loader):
                for k in data.keys():
                    data[k] = data[k].float().to(self.device)

                # A-NeRF 모델의 예측 값 가져오기
                ret = self.model(data)

                # Left, Right 채널로부터 스테레오 오디오 신호 결합
                prd_left_wav = ret["left"]  # 왼쪽 채널 (예측 신호)
                prd_right_wav = ret["right"]  # 오른쪽 채널 (예측 신호)
                prd_stereo_wav = torch.stack([prd_left_wav, prd_right_wav], dim=1)  # 스테레오 신호 결합 [B, 2, T]

                # Ground Truth 스테레오 신호 가져오기
                gt_stereo_wav = data["gt_data"]  # Ground truth 스테레오 신호 [B, 2, T]

                # 메트릭 계산 (T60, EDT, C50)
                metrics = self.evaluator.get_full_metrics(
                    prd_stereo_wav.cpu().numpy(),  # 예측 스테레오 오디오 신호
                    gt_stereo_wav.cpu().numpy(),   # Ground truth 스테레오 신호
                    gt_stereo_wav.cpu().numpy(),   # Ground truth IR
                    prd_stereo_wav.cpu().numpy(),  # 예측 IR
                    gt_stereo_wav.cpu().numpy(),   # Ground truth IR
                    log_prd=None, log_gt=None      # 로그 값은 사용하지 않음
                )

                t60_errors.append(metrics['audio_T60_mean_error'])
                edt_values.append(metrics['audio_EDT'])
                c50_values.append(metrics['audio_C50'])

                # tqdm 업데이트
                t.update()
            t.close()

        avg_t60_error = np.mean(t60_errors)
        avg_edt = np.mean(edt_values)
        avg_c50 = np.mean(c50_values)

        print(f"T60: {avg_t60_error}, C50: {avg_c50}, EDT: {avg_edt}")

        return

    


    def save_ckpt(self):
        try:
            state_dict = self.model.module.state_dict()  # remove prefix of multi GPUs
        except AttributeError:
            state_dict = self.model.state_dict()
        
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        torch.save({
                'epoch': self.epoch,
                'iter': self.iter_count,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict()},
                os.path.join(self.log_dir, f"{self.epoch}.pth"))
