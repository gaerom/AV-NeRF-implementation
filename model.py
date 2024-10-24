import math
import einops
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ANeRF(nn.Module):
    def __init__(self,
                 conv=False,
                 freq_num=257,
                 time_num=173,
                 intermediate_ch=128,
                 p=0):
        super(ANeRF, self).__init__()
        self.conv = conv
        self.freq_num = freq_num
        self.pos_embedder = embedding_module_log(num_freqs=10, ch_dim=1) # position (x, y)
        self.freq_embedder = embedding_module_log(num_freqs=10, ch_dim=1) # frequency f
        ### 63 -> 128-dim
        self.query_prj = nn.Sequential(nn.Linear(42 + 21, intermediate_ch), nn.ReLU(inplace=True)) # embedding position, frequency -> 결합해서 query로 변환 (해당 위치에서 어떤 주파수의 소리가 들릴지 querying)
        self.mix_mlp = MLPwSkip(intermediate_ch, intermediate_ch)
        self.mix_prj = nn.Linear(intermediate_ch, 1)
        self.ori_embedder = Embedding(4, 4, intermediate_ch)
        self.diff_mlp = MLPwSkip(intermediate_ch, intermediate_ch)
        self.diff_prj = nn.Linear(intermediate_ch, 1)
        self.av_mlp = nn.Sequential(nn.Linear(2048, 512), # rgb + depth feature -> A-NeRF에 통합하려고 -> 2048로 수정 ?
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, intermediate_ch),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(intermediate_ch, intermediate_ch))
        if self.conv:
            self.post_process = nn.Sequential(nn.Conv2d(4, 16, 7, 1, 3),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 16, 3, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 1, 3, 1, 1))
        
        self.p = p
        
    def forward(self, x):
        # x {"pos", "ori", "depth", "rgb"}
        B = x["pos"].shape[0]
        pos = self.pos_embedder(x["pos"]) # [B, 42], position: 42 dim
        pos = mydropout(pos, p=self.p, training=self.training)
        freq = torch.linspace(-0.99, 0.99, self.freq_num, device=x["pos"].device).unsqueeze(1) # [F, 1]
        freq = self.freq_embedder(freq) # [F, 21] # frequency: 21 dim

        pos = einops.repeat(pos, "b c -> b f c", f=self.freq_num)
        freq = einops.repeat(freq, "f c -> b f c", b=B)
        query = torch.cat([pos, freq], dim=2) # [B, F, ?]
        query = self.query_prj(query) # [B, F, ?]


        ### llava에서 뽑아낸 feature concat 하려면 여기서 해야 할 듯 -> x["rgb"], x["depth"] 차원도 확인해보기
        ### x["rgb"]: [32, 512] / x["depth"]: [32, 512]
        """
        x["text_embedding"] load 하기 전에 차원 [B, 1024] 로 맞춰야 함.
        
        """
        # print('with llava text embedding..')
        # text_embedding_batch = x["text_embedding"]
        v_feats = torch.cat([x["rgb"], x["depth"]], dim=1) # [B, 1024]

        ### llava feature와 concat
        # v_feats_t = torch.cat([v_feats, text_embedding_batch], dim=1)



        if self.training:
            noise = torch.randn_like(v_feats) * 0.1 
            v_feats = v_feats + noise
        v_feats = self.av_mlp(v_feats) # [B, ?] # MLP 통과한 visual feature
        v_feats = mydropout(v_feats, p=self.p, training=self.training)
        v_feats = einops.repeat(v_feats, "b c -> b 1 c") # frequency query 차원이랑 맞춰주기 위해?

        # predict mix mask
        feats = self.mix_mlp(query + v_feats)   ### embedding 처리 된 position, frequency -> query로 결합 -> (v_feats + query)가 1st MLP 통과 => feats 생성 
        mask_mix = self.mix_prj(feats).squeeze(-1) # [B, F], 1-dim -> mask, 마지막 dim 제거

        # predict diff mask
        ori = self.ori_embedder(x["ori"])
        ori = mydropout(ori, p=self.p, training=self.training)
        feats = self.diff_mlp(feats, ori) # 1st mlp feature + ori -> 방향 정보를 고려
        mask_diff = self.diff_prj(feats).squeeze(-1) # [B, F]
        mask_diff = torch.sigmoid(mask_diff) * 2 - 1 # [-1, 0, 1]

        time_dim = x["mag_sc"].shape[1]
        mask_mix = einops.repeat(mask_mix, "b f -> b t f", t=time_dim) # STFT한 magnitude랑 결합해야 하니까 차원 변경
        mask_diff = einops.repeat(mask_diff, "b f -> b t f", t=time_dim) 
        
        ### IID는 반영, ITD는 X (ITD 계산을 위해서는 time delay 반영)
        reconstr_mono = x["mag_sc"] * mask_mix # [B, T, F], s_m 생성하는 부분
        reconstr_diff = reconstr_mono * mask_diff # [B, T, F], s_d 생성하는 부분
        
        ### 만약 time delay를 반영한다면 
        """
        1. sample 단위로 time delay 계산 (sr=22050)
        2. (소리가 왼쪽에서 오는 경우) reconstr_left는 그대로, reconstr_right에는 time delay를 추가하는 방식
        -> 소리 방향 정보에 따른(x["ori"]) 방향 정보에 따라 ITD 계산 -> 한쪽 귀에 time delay를 추가하는 방식
        """
        reconstr_left = reconstr_mono + reconstr_diff # s_m + s_d 
        reconstr_right = reconstr_mono - reconstr_diff # s_m - s_d
        
        if self.conv:
            left_input = torch.stack([x["mag_sc"], mask_mix, mask_diff, reconstr_left], dim=1) # [B, 4, T, F]
            right_input = torch.stack([x["mag_sc"], mask_mix, -mask_diff, reconstr_right], dim=1) # [B, 4, T, F]
            left_output = self.post_process(left_input).squeeze(1) # Conv2D 
            right_output = self.post_process(right_input).squeeze(1) # Conv2D
            reconstr_left = reconstr_left + left_output
            reconstr_right = reconstr_right + right_output
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)
        else:
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)

        return {"mask_mix": mask_mix, 
                "mask_diff": mask_diff,
                "reconstr_mono": reconstr_mono,
                "reconstr": reconstr} # ISTFT에 들어가는 input?

class Embedding(nn.Module):
    def __init__(self, num_layer, num_embed, ch):
        super().__init__()
        self.embeds = nn.Parameter(torch.randn(num_embed, num_layer, ch) / math.sqrt(ch), requires_grad=True)
        self.num_embed = num_embed
    
    # linear interpolation
    def forward(self, ori):
        embeds = torch.cat([self.embeds[-1:], self.embeds, self.embeds[:1]], dim=0)
        ori = (ori + 1) / 2 * self.num_embed
        t_value = torch.arange(-1, self.num_embed + 1, device=ori.device)
        right_idx = torch.searchsorted(t_value, ori, right=False)
        left_idx = right_idx - 1

        left_dis = ori - t_value[left_idx]
        right_dis = t_value[right_idx] - ori
        left_dis = torch.clamp(left_dis, 0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1]
        right_dis = torch.clamp(right_dis, 0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1]

        left_embed = embeds[left_idx] # [B, l, c]
        right_embed = embeds[right_idx] # [B, l, c]

        output = left_embed * right_dis + right_embed * left_dis
        return output # [B, l, c]


class MLPwSkip(nn.Module):
    def __init__(self,
                 in_ch,
                 intermediate_ch=256,
                 layer_num=4,
                 ):
        super().__init__()
        self.residual_layer = nn.Linear(in_ch, intermediate_ch)
        self.layers = nn.ModuleList()
        for layer_idx in range(layer_num):
            in_ch_ = in_ch if layer_idx == 0 else intermediate_ch
            out_ch_ = intermediate_ch
            self.layers.append(nn.Sequential(nn.Linear(in_ch_, out_ch_),
                                             nn.ReLU(inplace=True)))

    def forward(self, x, embed=None):
        residual = self.residual_layer(x)
        for layer_idx in range(len(self.layers)):
            if embed is not None:
                # embed [B, l, c]
                x = self.layers[layer_idx](x) + embed[:, layer_idx].unsqueeze(1)
            else:
                x = self.layers[layer_idx](x)
            if layer_idx == len(self.layers) // 2 - 1:
                x = x + residual
        return x

class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=-1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)

def mydropout(tensor, p=0.5, training=True):
    if not training or p == 0:
        return tensor
    else:
        batch_size = tensor.shape[0]
        random_tensor = torch.rand(batch_size, device=tensor.device)
        new_tensor = [torch.zeros_like(tensor[i]) if random_tensor[i] <= p else tensor[i] for i in range(batch_size)]
        new_tensor = torch.stack(new_tensor, dim=0) # [B, ...]
        return new_tensor