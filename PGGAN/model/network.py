import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layer import NormalizationLayer, EqualizedLinear, EqualizedConv2d, Upsampling, Downsampling


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
        self.toRGB_layers.append(EqualizedConv2d(self.scale_channels[-1],
                                                 self.dim_output,
                                                 1))
    
    def SetAlpha(self, new_alpha):
        self.alpha = new_alpha
    
    def forward(self, x): # x.size() == (batch_size, 512, 1, 1)
        x = x.view(x.size()[0], -1)
        x = self.normalization_layer(x)
        x = self.fc_layer(x)
        x = self.activation_fn(x)
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalization_layer(x)

        for scale, scale_layers in enumerate(self.scale_layers):
            if scale:
                x = Upsampling(x)
            for scale_layer in scale_layers:
                x = scale_layer(x)
                x = self.activation_fn(x)
                x = self.normalization_layer(x)
            if scale == len(self.scale_layers) - 2 and self.alpha > 0.0:
                y = self.toRGB_layers[-2](x)
                y = Upsampling(x)
        
        x = self.toRGB_layers[-1](x)
        if self.alpha > 0.0:
            x = self.alpha * y + (1 - self.alpha) * x
        return x


class DNet(nn.Module):
    def __init__(self, channel_scale_0=512, dim_input=3, leakyReLU_slope=0.2):
        super(DNet, self).__init__()
        self.channel_scale_0 = channel_scale_0 # Dirty, find a better name
        self.dim_input       = dim_input
        self.alpha           = 0.0
        self.activation_fn   = nn.LeakyReLU(leakyReLU_slope)
        self.fc_layer        = EqualizedLinear(16 * self.channel_scale_0, self.channel_scale_0)
        self.decision_layer  = EqualizedLinear(self.channel_scale_0, 1)

        # channel per scale
        self.scale_channels = [self.channel_scale_0]

        # layer by scale
        self.scale_layers = nn.ModuleList()
        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(EqualizedConv2d(self.channel_scale_0,
                                                     self.channel_scale_0,
                                                     3,
                                                     padding=1))
        
        # fromRGB layer by scale
        self.fromRGB_layers = nn.ModuleList()
        self.fromRGB_layers.append(EqualizedConv2d(self.dim_input,
                                                   self.channel_scale_0,
                                                   1))
    
    def AddScale(self, new_channel):
        # update scale_channels
        self.scale_channels.append(new_channel)

        # update scale_layers
        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(EqualizedConv2d(self.scale_channels[-1],
                                                     self.scale_channels[-1],
                                                     3,
                                                     padding=1))
        self.scale_layers[-1].append(EqualizedConv2d(self.scale_channels[-1],
                                                     self.scale_channels[-2],
                                                     3,
                                                     padding=1))
        
        # update fromRGB_layers
        self.fromRGB_layers.append(EqualizedConv2d(self.dim_input,
                                                   self.scale_channels[-1],
                                                   1))
    
    def SetAlpha(self, new_alpha):
        self.alpha = new_alpha
    
    def forward(self, x): # x.size() == (batch_size, 3, 2^(n+2), 2^(n+2))
        if self.alpha > 0.0:
            y = Downsampling(x)
            y = self.fromRGB_layers[-2]
        
        x = self.fromRGB_layers[-1](x)
        x = self.activation_fn(x)

        for i, scale_layers in enumerate(reversed(self.scale_layers)):
            for scale_layer in scale_layers:
                x = scale_layer(x)
                x = self.activation_fn(x)
            if i != len(self.scale_layers) - 1:
                x = Downsampling(x)
            if i == 0 and self.alpha > 0.0:
                x = self.alpha * y + (1 - self.alpha) * x
        
        x = x.view(x.size()[0], -1)
        x = self.fc_layer(x)
        x = self.activation_fn(x)
        x = self.decision_layer(x)
        return x.view(-1)