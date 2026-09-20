import torch
print("版本:", torch.__version__)
print("可用设备:", "CPU")
x = torch.rand(3, 3)
print(x)
