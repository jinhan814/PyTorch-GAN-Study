import torch
## 실험용 입니다

# x = torch.randint(10,size=(1,4,2,2))
# print(x)
# print(x.size())

# factor =2
# s = x.size()
# x = x.view(-1, s[1], s[2], 1, s[3], 1)  # (-1, 4, 2, 1, 2, 1)
# print(x.size())
# # print(x)
# x = x.expand(-1, s[1], s[2], factor, s[3], factor) # (-1, 4,2,2,2,2)
# print(x.size())
# # print(x)
# x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
# # x = x.view(-1, s[1], s[2] * factor, s[3] * factor)
# print(x.size())
# # print(x)

# x = torch.rand(,4,2,2)
# subGroupSize = 4

# size = x.size()
# subGroupSize = min(size[0], subGroupSize)
# if size[0] % subGroupSize != 0:
#     subGroupSize = size[0]
# G = int(size[0] / subGroupSize)

# print(subGroupSize,G)
# print(x)
# if subGroupSize > 1:
#     y = x.view(-1, subGroupSize, size[1], size[2], size[3])
#     print(y)
#     y = torch.var(y, 1)
#     print(y)
#     y = torch.sqrt(y + 1e-8)
#     print(y)
#     y = y.view(G, -1)
#     print(y)
#     y = torch.mean(y, 1).view(G, 1)
#     print(y)
#     y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
#     print(y)
#     y = y.expand(G, subGroupSize, -1, -1, -1)
#     print(y)
#     y = y.contiguous().view((-1, 1, size[2], size[3]))
# else:
#     y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
#

import torch
import torchvision
import cv2
x = torch.randint(10,size=(8,8,3))
x= torch.transpose(x,(2,0,1))
print(x.size())
x = torchvision.transforms.Resize((4,4))(x)
x = torch.transpose(x,(1,2,0))
print(x.size())