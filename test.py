import torch
import torch.nn as nn
import torch.functional as F

target = torch.randn((1, 16, 18))
m = nn.AdaptiveAvgPool2d(7)

new = m(target)

print(new)
print(new.size())