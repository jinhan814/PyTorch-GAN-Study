import os
import json
from PGGAN.utils import resizing
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import PGGAN
from dataset import IdolDataset
from torch.utils.data import DataLoader


def SeedEverything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def Train(args):
    SeedEverything(args['seed'])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    model = PGGAN(512, 3, 0.2)
    model.ToDevice(device)
    
    batch_size = args['batch_size']
    latent_vector_size = 512
    criterion = model.criterion

    train_dataset = IdolDataset(args['train_dir'])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    
    optimizer_D = model.OptD
    optimizer_G = model.OptG
    
    label_real  = torch.ones(batch_size, 1).to(device)
    label_fake  = torch.zeros(batch_size, 1).to(device)
    
    model.train()
    for scale in range(args['scales']):
        for epoch in range(args['epochs'][scale]):
            alpha = 1-epoch/args['epochs'][scale] if scale else 0
            model.SetAlpha(alpha)
            
            loss_D_per_epoch = 0
            loss_G_per_epoch = 0
            
            for batch_idx, img  in enumerate(train_loader):
                
                img = resizing(img, args['img_size'][scale])
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
    json_path = "./config.json"
    with open(json_path) as f:
        config_json = json.load(f)
        
    Train(config_json)