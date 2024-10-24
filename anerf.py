import math
import einops
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import GriffinLim # for predicted RIR

class ANeRF(nn.Module):
    def __init__(self,
                 visual=False,
                 relative=False,
                 intermediate_ch=256):
        super(ANeRF, self).__init__()
        self.visual = visual
        self.relative = relative
        self.pos_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
        self.register_buffer("times", 2*torch.arange(0, 22050)/22050 - 1.0) # [-1.0, 1.0] normalization, default: 15000
        self.time_embedder = embedding_module_log(num_freqs=10, ch_dim=1)

        # self.query_prj = nn.Sequential(nn.Linear(42 + 42 + 21, intermediate_ch), nn.ReLU(inplace=False)) # source, target, time
        self.query_prj = nn.Sequential(nn.Linear(63 + 63 + 63, intermediate_ch), nn.ReLU(inplace=False))
        self.mix_mlp = MLPwSkip(intermediate_ch, intermediate_ch)

        if relative:
            self.ori_embedder = Embedding(4, 4, intermediate_ch)
            self.diff_mlp = MLPwSkip(intermediate_ch, intermediate_ch)
        else: # False
            self.ori_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
            # self.diff_mlp = MLPwSkip(intermediate_ch + 21, intermediate_ch)
            self.diff_mlp = MLPwSkip(intermediate_ch + 63, intermediate_ch)

        self.diff_prj = nn.Linear(intermediate_ch, 1) # 왼쪽, 오른쪽 귀 사이 binaural 신호 차이

        # 왼쪽, 오른쪽 귀에 대한 embedding vector, random initialization
        self.left_embed = nn.Parameter(torch.randn(4, intermediate_ch) / math.sqrt(intermediate_ch), requires_grad=True)
        self.right_embed = nn.Parameter(torch.randn(4, intermediate_ch) / math.sqrt(intermediate_ch), requires_grad=True)

        if visual:
            self.av_mlp = nn.Sequential(nn.Linear(1024, 512),
                                        nn.ReLU(inplace=False),
                                        nn.Linear(512, intermediate_ch),
                                        nn.ReLU(inplace=False),
                                        nn.Linear(intermediate_ch, intermediate_ch))
            
    ### x[""] -> SoundSpaces dataset에서 뭐에 해당하는지만 알면 되지 않을까  
    def forward(self, x):
        # RWAVS x {"pos", ori, rgb, depth, mag_sc}
        # SoundSpaces x {"source", "target(listener)", "ori(listener head)", "wav_len", "vision"}
        # source: source_pose, target: mic_pose, ori: rot, wav_len: X

        B = x["source_pose"].shape[0]
        x["rot"] = x["rot"].float()


        source = self.pos_embedder(x["source_pose"]) # [B, 42] / [B, 63]
        target = self.pos_embedder(x["mic_pose"]) # [B, 42] / [B, 63]
        # max_len = x["wav_len"].max()
        max_len = 22050
        time = self.times[:max_len].unsqueeze(1) # [T, 1] / [15000, 1]
        time = self.time_embedder(time) # [T, 21] / [15000, 21]
        

        source = einops.repeat(source, "b c -> b t c", t=max_len) # [1, 22050, 63]
        target = einops.repeat(target, "b c -> b t c", t=max_len) # [1, 22050, 63]
        time = einops.repeat(time, "t c -> b t c", b=B) # [1, 22050, 21]

        time_padded = torch.nn.functional.pad(time, (0, 63 - time.shape[2])) # add [1, 22050, 63]
    
        query = torch.cat([source, target, time_padded], dim=2) # [B, t, ?] / [B, 22050, 189]
        query = self.query_prj(query) # [B, t, ?] / [1, 22050, 256]
        

        if self.visual:
            v_feats = x["vision"] # [B, 1024]
            v_feats = self.av_mlp(v_feats) # [B, ?]
            v_feats = einops.repeat(v_feats, "b c -> b l c", l=4)

        if self.visual:
            feats_in = self.mix_mlp(query, v_feats) # (source, target, time) + visual feature
        else: # 일단 vision 없이
            feats_in = self.mix_mlp(query) # [1, 22050, 256]
        
        #ori = self.ori_embedder(x["ori"])
        #feats = self.diff_mlp(feats_in, v_feats + self.left_embed.unsqueeze(0))
        #prd_left_wav = self.diff_prj(feats).squeeze(-1) # [B, T]
        channel_embed = einops.repeat(self.left_embed, "l c -> b l c", b=B) ### left
        if self.relative:
            ori = self.ori_embedder(x["rot"])
            if self.visual:
                feats = self.diff_mlp(feats_in, ori + v_feats + channel_embed)
            else:
                feats = self.diff_mlp(feats_in, ori + channel_embed)
        else: # False
            # ori = x["rot"].unsqueeze(1) # [B, 1]
            ori = x["rot"]
            ori = self.ori_embedder(ori) # [B, 21] / [B, 63]
            ori = einops.repeat(ori, "b c -> b t c", t=max_len) # [1, 22050, 63]
            feats = torch.cat([feats_in, ori], dim=2)
            if self.visual:
                feats = self.diff_mlp(feats, v_feats + channel_embed)
            else:
                feats = self.diff_mlp(feats, channel_embed)
        prd_left_wav = self.diff_prj(feats).squeeze(-1) # [B, T], 

        channel_embed = einops.repeat(self.right_embed, "l c -> b l c", b=B) ### right
        if self.relative:
            ori = self.ori_embedder(x["ori"])
            if self.visual:
                feats = self.diff_mlp(feats_in, ori + v_feats + channel_embed)
            else:
                feats = self.diff_mlp(feats_in, ori + channel_embed)
        else:
            # ori = x["rot"].unsqueeze(1) # [B, 1] / [1, 1, 3]
            ori = x["rot"] 
            ori = self.ori_embedder(ori) # [B, 21] / [B, 63]
            ori = einops.repeat(ori, "b c -> b t c", t=max_len) # # [1, 22050, 63]
            feats = torch.cat([feats_in, ori], dim=2) # [1, 22050, 319]
            if self.visual:
                feats = self.diff_mlp(feats, v_feats + channel_embed)
            else:
                feats = self.diff_mlp(feats, channel_embed)
        prd_right_wav = self.diff_prj(feats).squeeze(-1) # [B, T] / [1,]

        return {"left": prd_left_wav,
                "right": prd_right_wav}
        
        
"""
griffin_lim = GriffinLim(n_fft=512, win_length=512, hop_length=128)

left_wav = griffin_lim(prd_left_wav)
right_wav = griffin_lim(prd_right_wav)

prd_rir = torch.stack([left_wav, right_wav], dim=0) # 이게 metric 측정 시 pred

"""

class Embedding(nn.Module):
    def __init__(self, num_layer, num_embed, ch):
        super().__init__()
        self.embeds = nn.Parameter(torch.randn(num_embed, num_layer, ch) / math.sqrt(ch), requires_grad=True)
        self.num_embed = num_embed
    
    def forward(self, ori):
        embeds = torch.cat([self.embeds[-1:], self.embeds, self.embeds[:1]], dim=0)
        ori = (ori + 1) / 2 * self.num_embed
        t_value = torch.arange(-1, self.num_embed + 1, device=ori.device)
        right_idx = torch.searchsorted(t_value, ori, right=False)
        left_idx = right_idx - 1

        left_embed = embeds[left_idx] # [B, l, c]
        right_embed = embeds[right_idx] # [B, l, c]

        left_dis = ori - t_value[left_idx]
        right_dis = t_value[right_idx] - ori
        left_dis = torch.clamp(left_dis, 0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1]
        right_dis = torch.clamp(right_dis, 0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1]

        output = left_embed * right_dis + right_embed * left_dis
        return output # [B, l, c]


class MLPwSkip(nn.Module):
    def __init__(self,
                 in_ch,
                 intermediate_ch=256,
                 layer_num=4,
                 ):
        super().__init__()
        #self.residual_layer = nn.Linear(in_ch, intermediate_ch)
        self.layers = nn.ModuleList()
        for layer_idx in range(layer_num):
            in_ch_ = in_ch if layer_idx == 0 else intermediate_ch
            out_ch_ = intermediate_ch
            self.layers.append(nn.Sequential(nn.Linear(in_ch_, out_ch_),
                                             nn.ReLU(inplace=False)))

    def forward(self, x, embed=None):
        #residual = self.residual_layer(x)
        #print("mix mlp", x.shape, residual.shape)
        for layer_idx in range(len(self.layers)):
            if embed is not None:
                # embed [B, l, c]
                #print(layer_idx, x.shape, embed[:, layer_idx].unsqueeze(1).shape)
                x = self.layers[layer_idx](x) + embed[:, layer_idx].unsqueeze(1)
            else:
                x = self.layers[layer_idx](x)
            #if layer_idx == len(self.layers) // 2 - 1:
            #    x = x + residual
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