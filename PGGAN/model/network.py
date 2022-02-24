import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_layer import NormalizationLayer, EqualizedLinear, EqualizedConv2d, Upsampling

class GNet(nn.Module):
    def __init__(self,
                 dimLatent=512,
                 leakyReluLeak=0.2,
                 dimOutput=3):
        super(GNet, self).__init__()
        '''
        things to do

        1. latent vector(512 * 1 * 1) -> (512 * 4 * 4) (fc + view)
        2. scale 별 upsample + conv + @(normalization, lReLU, ...)
        3. toRGB 적용 + alpha 비율대로 합치기
        '''
        self.dimLatent = dimLatent
        self.leakyRelu = nn.LeakyReLU(leakyReluLeak)
        self.dimOutput = dimOutput
        self.alpha = 0
        self.normalizationLayer = NormalizationLayer()
        self.formatLayer = EqualizedLinear(self.dimLatent,
                                           16 * self.dimLatent)
        
        # scale별 layer block
        self.scaleLayers = nn.ModuleList()
        self.scaleLayers.append(nn.ModuleList([
            EqualizedConv2d(dimLatent,
                            dimLatent,
                            3,
                            padding=1),
            self.leakyRelu,
            self.normalizationLayer
        ]))

        # scale별 toRGB layer
        self.toRGBLayer = nn.ModuleList()
    
    def addScale(self, depthNewScale):
        pass
    
    def forward(self, x):
        x = self.normalizationLayer(x)
        x = x.view(-1, torch.prod(x.size()[1:]))
        x = self.leakyRelu(self.formatLayer(x))
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalizationLayer(x)

        for scale, scaleLayer in enumerate(self.scaleLayers):
            x = scaleLayer(x)
            if scale == len(self.scaleLayers) - 2 and alpha > 0:
                y = Upsampling(x)
                y = self.toRGBLayers[-1](x)
        
        x = self.toRGBLayers[-1](x)
        if self.alpha > 0:
            x = self.alpha * y + (1 - self.alpha) * x
        return x


class discriminator(nn.module):
    def __init__(self,):
        pass
    def initLayer():
        pass
    def add_sacle():
        pass
    def foward():
        pass

