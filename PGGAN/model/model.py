import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network import GNet, DNet

from loss import LeastSquare

class PGGAN():
    def __init__(self,
                 channel_scale_0=512,
                 dim_output=3,
                 leakyReLU_slope=0.2,
                 learningRate=0.001):
        self.GNet = GNet(channel_scale_0, dim_output, leakyReLU_slope)
        self.DNet = DNet(channel_scale_0, dim_output, leakyReLU_slope)
        self.criterion = LeastSquare
        self.OptG = torch.optim.Adam(self.GNet.parameters(), lr=learningRate, betas=[0,0.99],eps=10e-9,weight_decay=0.999)
        self.OptD = torch.optim.Adam(self.DNet.parameters(), lr=learningRate, betas=[0,0.99],eps=10e-9,weight_decay=0.999)
    
    def AddScale(self, new_channel):
        self.GNet.AddScale(new_channel)
        self.DNet.AddScale(new_channel)
    
    def SetAlpha(self, new_alpha):
        self.GNet.SetAlpha(new_alpha)
        self.DNet.SetAlpha(new_alpha)
    
    def ToDevice(self,device):
        self.GNet.to(device)
        self.DNet.to(device)
        
    def train(self):
        self.GNet.train()
        self.DNet.train()

    def val(self):
        self.GNet.val()
        self.DNet.val()
