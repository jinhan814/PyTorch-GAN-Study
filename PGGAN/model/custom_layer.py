import torch
import torch.nn as nn

import math
from numpy import prod

def getLayerNormalizationFactor(x):
    
    size = x.weight.size()
    fan_in = prod(size[1:])

    return math.sqrt(2.0 / fan_in)

class ConstrainedLayer(nn.Module):
    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 InitBiasToZero=True):
        
        super(ConstrainedLayer,self).__init__()
        
        self.module = module
        self.equalized = equalized
        
        self.module.weight.data.normal_(0,1)   ## normal_함수: 평균, 표준편차로 정규화
        self.module.weight.data /= lrMul
        self.weight = getLayerNormalizationFactor(self.module) *lrMul
        
        self. module.bias.data.fill_(0)
        
    def forward(self, x):
        
        x = self.module(x)
        x *= self.weight
        
        return x
    
    
class EqualizedConv2d(ConstrainedLayer):
    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 kernelSize,
                 padding=0,
                 bias=True,
                 **kwargs):
        
        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(nChannelsPrevious,
                                            nChannels,
                                            kernelSize,
                                            padding=padding,
                                            bias=bias),
                                  **kwargs
                                  )
        
class EqualizedLinear(ConstrainedLayer):
    
    def __init__(self, 
                 nChannelsPrevious,
                 nChannels,
                 bias=True,
                 **kwargs):
        
        ConstrainedLayer.__init__(self,
                                  nn.Linear(nChannelsPrevious,
                                            nChannels,
                                            bias=bias),
                                  **kwargs
                                  )
        
        
        
        