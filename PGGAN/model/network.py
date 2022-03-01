import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_layer import NormalizationLayer, EqualizedLinear, EqualizedConv2d, Upsampling, Downsampling


class GNet(nn.Module):
    def __init__(self, dim_latent=512, dim_output=3, leakyReLU_slope=0.2):
        super(GNet, self).__init__()
        self.dim_latent          = dim_latent
        self.dim_output          = dim_output
        self.alpha               = 0.0
        self.activation_fn       = nn.LeakyReLU(leakyReLU_slope)
        self.normalization_layer = NormalizationLayer()
        self.fc_layer            = EqualizedLinear(self.dim_latent, 16 * self.dim_latent)
        
        # channel per scale
        self.scale_channels = [self.dim_latent]
        
        # layer by scale
        self.scale_layers = nn.ModuleList()
        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(EqualizedConv2d(self.dim_latent,
                                                     self.dim_latent,
                                                     3,
                                                     padding=1))
        
        # toRGB layer by scale
        self.toRGB_layers = nn.ModuleList()
        self.toRGB_layers.append(EqualizedConv2d(self.dim_latent,
                                                 self.dim_output,
                                                 1))
    
    def AddScale(self, new_channel):
        # update scale_channels
        self.scale_channels.append(new_channel)

        # update scale_layers
        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(EqualziedConv2d(self.scale_channels[-2],
                                                     self.scale_channels[-1],
                                                     3,
                                                     padding=1))
        self.scale_layers[-1].append(EqualziedConv2d(self.scale_channels[-1],
                                                     self.scale_channels[-1],
                                                     3,
                                                     padding=1))
        
        # update toRGB_layers
        self.toRGB_layers.append(EqualizedConv2d(self.sclae_channes[-1],
                                                 self.dim_output,
                                                 1))
    
    def forward(self, x):
        x = self.normalization_layer(x)
        x = x.view(-1, torch.prod(x.size()[1:]))
        x = self.fc_layer(x)
        x = self.activation_fn(x)
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalization_layer(x)

        for scale, scale_layer in enumerate(self.scale_layers):
            x = scale_layer(x)
            if scale == len(self.scale_layers) - 2 and self.alpha > 0.0:
                y = self.toRGB_layers[-2](x)
                y = Upsampling(x)
        
        x = self.toRGB_layers[-1](x)
        if self.alpha > 0.0:
            x = self.alpha * y + (1 - self.alpha) * x

        return x


class DNet(nn.module):
    def __init__(self,
                 depthScale0=512,
                 leakyReluLeak=0.2,
                 dimInput=3):
        
        super(DNet, self).__init__()

        self.depthscale0 = depthScale0
        self.leakyRelu = nn.LeakyReLU(leakyReluLeak)
        self.dimInput = dimInput

        self.alpha = 0
        self.decisionlayer = EqualizedLinear(self.depthScale0,1)
        
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
        
    def add_sacle():
        pass
    
    def forward(self, x):
        pass