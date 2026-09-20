import torch
x = torch.tensor(3.0, requires_grad=True)
print(x)            # tensor(3., requires_grad=True) 
y = x ** 3          # 这时账本记下：y 由 x 做"三次方"得到
y.backward()        # 自动求 dy/dx，把结果存进 x.grad
print(x.grad)       # tensor(27.)
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

y = a * b + b ** 2
y.backward()

print(a.grad)   
print(b.grad)   
import torch

w = torch.tensor(2.0, requires_grad=True)
lr = 0.1

for i in range(10):
    loss = (w - 5) ** 2
    loss.backward()          # 算梯度   
    print(f"step {i}: w={w.item():.3f}, grad={w.grad.item():.3f}, loss={loss.item():.3f}")  
    with torch.no_grad():    # 更新参数时不要记进计算图
         w -= lr * w.grad
         w.grad.zero_()           # 梯度清零，否则会累加（前面讲过的坑）