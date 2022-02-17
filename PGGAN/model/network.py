import torch
import torch.nn as nn

class GNet(nn.Module):
    def __init__(self,
                 dimLatent,
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 normalization=True,
                 generationActivation=None,
                 dimOutput=3,
                 equalizedlR=True):