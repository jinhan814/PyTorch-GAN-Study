import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_layer import NormalizationLayer, EqualizedLinear, EqualizedConv2d, Upsampling, Downsampling

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
            EqualizedConv2d(self.dimLatent,
                            self.dimLatent,
                            3,
                            padding=1),
            self.leakyRelu,
            self.normalizationLayer
        ]))

        # scale별 toRGB layer
        self.toRGBLayer = nn.ModuleList()
        
        # scale depth 리스트
        self.scaledepths = [dimLatent]
    
    def addScale(self, depthNewScale):
        
        depthLastScale = self.scalesDepth[-1]

        self.scalesDepth.append(depthNewScale)

        self.scaleLayers.append(nn.ModuleList())

        self.scaleLayers[-1].append(EqualizedConv2d(depthLastScale,
                                                    depthNewScale,
                                                    3,
                                                    padding=1))
        self.scaleLayers[-1].append(self.leakyRelu)
        self.scaleLayers[-1].append(self.normalizationLayer)
        
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale, depthNewScale,
                                                    3, padding=1))
        self.scaleLayers[-1].append(self.leakyRelu)
        self.scaleLayers[-1].append(self.normalizationLayer)

        self.toRGBLayer.append(EqualizedConv2d(depthNewScale,
                                                self.dimOutput,
                                                1))
    
    def forward(self, x):
        x = self.normalizationLayer(x)
        x = x.view(-1, torch.prod(x.size()[1:]))
        x = self.leakyRelu(self.formatLayer(x))
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalizationLayer(x)

        for scale, scaleLayer in enumerate(self.scaleLayers):
            x = scaleLayer(x)
            if scale == len(self.scaleLayers) - 2 and self.alpha > 0:
                y = Upsampling(x)
                y = self.toRGBLayers[-1](x)
        
        x = self.toRGBLayer[-1](x)
        if self.alpha > 0:
            x = self.alpha * y + (1 - self.alpha) * x
        return x


class DNet(nn.module):
    def __init__(self,depthScale0,
                 leakyReluLeak=0.2,
                 dimInput=3):
        self.depthscale0 = depthScale0
        self.dimInput = dimInput
        self.leakyRelu = nn.LeakyReLU(leakyReluLeak)
        self.alpha = 0
        self.decisionlayer = EqualizedLinear(self.depthScale0,1)   # depthscale0 = 512
        
        self.groupscalezero = EqualizedLinear(depthScale0 *4 *4,
                                              depthScale0)
        
        self.scaleLayers = nn.ModuleList()
        self.scaleLayers.append(nn.ModuleList([
            EqualizedConv2d(self.depthScale0,
                            self.depthScale0,
                            3,
                            padding=1),
            self.leakyRelu
        ]))
        
        self.fromRGBlayer = nn.ModuleList()
        self.fromRGBlayer.append(EqualizedConv2d(self.dimInput,
                                                 self.depthScale0))
        
        '''
        우리는 오늘 장렬히 일주일을 마무리하겠다.
        '''
        
    def add_sacle():
        pass
    def foward(self, x):
        
        if self.alpha > 0    ##### 모르겠당
            y = Downsampling(x)
            y = self.fromRGBlayer[](y)
            
        
        pass

