import numpy
import torch
import torch.nn.functional as F

def resizing(img,s):
    img  = F.interpolate(img,size=(s,s))
        
        
        