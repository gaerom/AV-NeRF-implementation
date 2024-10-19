import os
import PIL
import glob
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import soundfile as sf

import torch
from torchvision.models import resnet18, ResNet18_Weights

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default="/mnt/storage1/AV-NeRF/release")
    parser.add_argument('--split', type=str, default="val")
    parser.add_argument('--save-dir', type=str, default="./results_GT")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # ResNet-18 Encoder
    weights = ResNet18_Weights.DEFAULT # torchvision
    original_resnet = resnet18(weights=weights, progress=False).eval() # encoder 이 부분 수정해봐야 함.
    layers = list(original_resnet.children())[:-1] # fully connected layer (for classification) -> 제거
    model = torch.nn.Sequential(*layers) # 나머지 layer -> nn.Sequential 처리
    model.to("cuda:0")
    transforms = weights.transforms() # resnet18에 맞게 변환
    

    rgb_list = sorted(glob.glob(os.path.join(args.data_dir, "13/frames_val/*.png"))) # rendered RGB 
    print(len(rgb_list))
    depth_list = sorted(glob.glob(os.path.join(args.data_dir, "13/frames_val_depth/*.png"))) # depth 
    print(len(depth_list))
    # mask_list = sorted(glob.glob(os.path.join(args.data_dir, "output_json_13_eval/test/ours_30000/masks/*.png"))) # segmented_image 
    # print(len(mask_list))
    
    features = {"rgb": [],
                "depth": []}
    # features = {"rgb": [],
    #             "mask": []}
    # features = {"depth": [],
    #             "mask": []}
    
    
    for rgb in rgb_list:
        rgb = PIL.Image.open(rgb).convert('RGB')
        rgb = transforms(rgb).unsqueeze(0) # [1, 3, h, w]
        with torch.no_grad(): feature = model(rgb.to("cuda:0")).squeeze().cpu().numpy()
        features["rgb"].append(feature)
        
    for depth in depth_list:
        depth = PIL.Image.open(depth).convert('RGB')
        depth = transforms(depth).unsqueeze(0) # [1, 3, h, w]
        # print(depth.shape)
        with torch.no_grad(): feature = model(depth.to("cuda:0")).squeeze().cpu().numpy()
        features["depth"].append(feature)
    
    # for mask in mask_list:
    #     mask = PIL.Image.open(mask).convert('RGB')
    #     mask = transforms(mask).unsqueeze(0) # [1, 3, h, w]
    #     # print(depth.shape)
    #     with torch.no_grad(): feature = model(mask.to("cuda:0")).squeeze().cpu().numpy()
    #     features["mask"].append(feature)
        
    pickle.dump(features, open(os.path.join(args.save_dir, f"feats_GT_{args.split}_13.pkl"), "wb"))