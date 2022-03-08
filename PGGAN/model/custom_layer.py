from ast import AsyncFunctionDef
import torch
import torch.nn as nn

import math
from numpy import prod

class NormalizationLayer(nn.Module):
    
    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())

def Upsampling(x, factor=2):
    # assert isinstance(factor, int) and factor >=1  ## 이게 있어야할까? 라는 궁금증
    
    # if factor == 1 :  ## 이것도 있어야 할까?
    #     return x  
    
    s =  x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    
    ## contiguous를 써주는 이유
    '''
    tensor를 다양한 함수(transpose,view)등을 이용하여 변형 시킬때 size와 stride는
    형태(순서)가 달라질 수 있으나 실제로 메모리상의 원소들의 위치는 바뀌지 않고 접근 인덱스만
    바뀐다. 따라서 추가로 변형을 할때는 그 메모리도 재할당해줄 필요가 있다. 
    '''
    return x

def Downsampling(x):
    return nn.functional.avg_pool2d(x, (2,2))


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
        
        
        
def MinibatchStddev(x,subGroupSize=4):

    size = x.size()
    subGroupSize = min(size[0], subGroupSize)
    if size[0] % subGroupSize != 0:
        subGroupSize = size[0]
    G = int(size[0] / subGroupSize)
    if subGroupSize > 1:
        y = x.view(-1, subGroupSize, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, subGroupSize, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    return torch.cat([x, y], dim=1)   