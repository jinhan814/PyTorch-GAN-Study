import os
import json
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

    # 2) DataLoader 정의
    train_dataset = IdolDataset(args['train_dir'])
    val_dataset = IdolDataset(args['val_dir'])
    train_loader = DataLoader(train_dataset, args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, args['batch_size'], shuffle=False)

    # 3) epoch 돌리면서 학습시키기
    for scale in range(args['scales']):
        model.train()
        for epoch in range(args['epochs'][scale]):
            # dataloader 보면서 통과시키고 역전파
            # alpha값 감소시키기
            alpha = 1-epoch/args['epochs'][scale] if scale else 0
            model.SetAlpha(alpha)
            
            
        model.AddScale(args['channels'][scale])


if __name__ == "__main__":
    json_path = ""
    with open(json_path) as f:
        config_json = json.load(f)
    
    Train(config_json)