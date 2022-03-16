import os
import json
from cv2 import transform
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import PGGAN
from dataset import IdolDataset
from torch.utils.data import DataLoader

"""
해야할 일

1. config 만들기

2. train 돌리기

-- 1) PGGAN 모델 정의

-- 2) DataLoader 정의

-- 3) epoch 돌리면서 학습시키기

-- 4) 중간 결과 wandb나 다른 곳에 저장하기
"""

def SeedEverything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# 일단 스켈레톤 코드만
def Train(args):
    SeedEverything(args['seed'])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # 1) PGGAN 모델 정의
    model = PGGAN(512, 3, 0.2)
    model.ToDevice(device)
    
    batch_size = args['batch_size']
    latent_vector_size = 512
    criterion = model.criterion

    # 2) DataLoader 정의
    train_dataset = IdolDataset(args['train_dir'])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = IdolDataset(args['val_dir'])
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    optimizer_D = model.OptD
    optimizer_G = model.OptG
    
    label_real  = torch.ones(batch_size, 1).to(device)
    label_fake  = torch.zeros(batch_size, 1).to(device)
    
    model.train()
    # 3) epoch 돌리면서 학습시키기
    for scale in range(args['scales']):

        
        for epoch in range(args['epochs'][scale]):
            # dataloader 보면서 통과시키고 역전파
            # alpha값 감소시키기
            alpha = 1-epoch/args['epochs'][scale] if scale else 0
            model.SetAlpha(alpha)
            
            loss_D_per_epoch = 0
            loss_G_per_epoch = 0
            
            for batch_idx, img  in enumerate(train_loader):
        
                img = img.to(device)
                z = torch.randn(batch_size, latent_vector_size).to(device)
                loss_D = criterion(model.DNet(img), label_real) + criterion(model.DNet(model.GNet(z)), label_fake)
                model.DNet.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                
                # z = torch.randn(batch_size, latent_vector_size).to(device)  확인해보기!
                loss_G = criterion(model.DNet(model.GNet(z)), label_real)
                model.GNet.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                
                loss_D_per_epoch += loss_D.item()
                loss_G_per_epoch += loss_G.item()
            
            loss_D_per_epoch = loss_D_per_epoch / (batch_idx+1)
            loss_G_per_epoch = loss_G_per_epoch / (batch_idx+1)
            
            print(f'Epoch: {epoch+1}/{args["epochs"][scale]}\t Loss_D: {loss_D_per_epoch:.6f}\t Loss_G: {loss_G_per_epoch:.6f}\t')
            
            
        model.AddScale(args['channels'][scale])


if __name__ == "__main__":
    json_path = ""
    with open(json_path) as f:
        config_json = json.load(f)
    
    Train(config_json)