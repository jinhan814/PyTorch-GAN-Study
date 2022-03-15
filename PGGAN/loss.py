import torch
import torch.nn as nn


def LeastSquare(x, y):
    return 0.5 * torch.mean((x - y) ** 2)