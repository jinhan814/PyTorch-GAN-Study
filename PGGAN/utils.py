import numpy
import torch
from torchvision import transforms

def resizing(img,s):
    img  = transforms.Resize(s)(img)
        