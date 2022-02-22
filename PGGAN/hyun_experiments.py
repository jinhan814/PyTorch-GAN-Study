import torch
## 실험용 입니다

x = torch.randint(10,size=(1,4,2,2))
print(x)
print(x.size())

factor =2
s = x.size()
x = x.view(-1, s[1], s[2], 1, s[3], 1)  # (-1, 4, 2, 1, 2, 1)
print(x.size())
# print(x)
x = x.expand(-1, s[1], s[2], factor, s[3], factor) # (-1, 4,2,2,2,2)
print(x.size())
# print(x)
x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
# x = x.view(-1, s[1], s[2] * factor, s[3] * factor)
print(x.size())
# print(x)