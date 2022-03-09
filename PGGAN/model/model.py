import torch
import torch.nn as nn
import torch.nn.functional as F
from network import GNet, DNet

class PGGAN():
    def __init__(self,
                 channel_scale_0=512,
                 dim_output=3,
                 leakyReLU_slope=0.2):
        self.GNet = GNet(channel_scale_0,
                         dim_output,
                         leakyReLU_slope)
        self.DNet = DNet(channel_scale_0,
                         dim_output,
                         leakyReLU_slope)
        self.OptG = None # WIP
        self.OptD = None # WIP
    
    def AddScale(self, new_channel):
        self.GNet.AddScale(new_channel)
        self.DNet.AddScale(new_channel)
    
    def SetAlpha(self, new_alpha):
        self.GNet.SetAlpha(new_alpha)
        self.DNet.SetAlpha(new_alpha)