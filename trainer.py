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
import torchaudio

import soundfile as sf

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
            ret = self.model(data)

            # Ground truth와 예측값을 STFT로 변환
            mag_bi_gt = torch.stft(data["wav_bi"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
            mag_bi_pred = torch.stft(ret["reconstr"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
            
            # 크기 정보만 사용해 L2 Loss 계산
            gt_magnitude = torch.abs(mag_bi_gt)
            pred_magnitude = torch.abs(mag_bi_pred)
            loss = F.mse_loss(pred_magnitude, gt_magnitude)  # L2 Loss

            # Griffin-Lim으로 시간 도메인 신호 복원 (저장 및 분석용)
            wav_pred_left = self.griffin_lim(ret["left"])
            wav_pred_right = self.griffin_lim(ret["right"])
            pred_rir = torch.stack([wav_pred_left, wav_pred_right], dim=1)  # [B, 2, T]

            # 최종 Loss 업데이트 및 최적화
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 학습 기록 저장
            self.writer.add_scalar("train/loss", loss, self.iter_count)
            t.update()
            self.iter_count += 1
        t.close()

    def eval(self, val_loader, save=False):
        self.model.eval()
        
        with torch.no_grad():
            t = tqdm(total=len(val_loader), desc=f"[EPOCH {self.epoch} EVAL]", leave=False)
            for data_idx, data in enumerate(val_loader):
                for k in data.keys():
                    data[k] = data[k].float().to(self.device)
                ret = self.model(data)

                # Ground truth와 예측값을 STFT로 변환
                mag_bi_gt = torch.stft(data["wav_bi"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
                mag_bi_pred = torch.stft(ret["reconstr"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)

                # 크기 정보만 사용해 L2 Loss 계산
                gt_magnitude = torch.abs(mag_bi_gt)
                pred_magnitude = torch.abs(mag_bi_pred)
                loss = F.mse_loss(pred_magnitude, gt_magnitude)  # L2 Loss

                # Griffin-Lim으로 시간 도메인 신호 복원
                wav_pred_left = self.griffin_lim(ret["left"])
                wav_pred_right = self.griffin_lim(ret["right"])
                pred_rir = torch.stack([wav_pred_left, wav_pred_right], dim=1)  # [B, 2, T]

                # T60, EDT, C50 메트릭 계산
                metrics = self.evaluator.get_full_metrics(
                    mag_bi_pred.cpu().numpy(), 
                    mag_bi_gt.cpu().numpy(),
                    data["wav_bi"].cpu().numpy(),  # 원래 ground truth IR
                    pred_rir.cpu().numpy(),         # 복원된 예측 IR
                    data["wav_bi"].cpu().numpy(),   # 원래 ground truth IR
                    log_prd=None, log_gt=None       # 로그 값은 사용하지 않음
                )

                # 메트릭 결과 출력
                print(f"T60 Mean Error: {metrics['audio_T60_mean_error']}, EDT: {metrics['audio_EDT']}, C50: {metrics['audio_C50']}")

                # 평가 결과 저장 및 리포트
                t.update()
            t.close()
        result = self.evaluator.report()
        if hasattr(self, "writer"):
            for k, v in result.items():
                self.writer.add_scalar(f"eval/{k}", v, self.epoch)
        return result

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
