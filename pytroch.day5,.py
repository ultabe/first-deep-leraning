import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# ---------- 1. 造数据 ----------
x0 = torch.randn(100, 2) + torch.tensor([-2.0, -2.0])
y0 = torch.zeros(100, 1)
x1 = torch.randn(100, 2) + torch.tensor([2.0, 2.0])
y1 = torch.ones(100, 1)

x = torch.cat([x0, x1], dim=0)      # (200, 2)
y = torch.cat([y0, y1], dim=0)      # (200, 1)

# ---------- 2. 模型 ----------
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)      

model = LogisticRegression()


loss_fn = nn.BCEWithLogitsLoss()                
optimizer = optim.SGD(model.parameters(), lr=0.1)


for epoch in range(200):
    y_logits = model(x)                  
    loss = loss_fn(y_logits, y)         

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        prob = torch.sigmoid(y_logits)
        pred = (prob > 0.5).float()
        acc = (pred == y).float().mean()
        print(f"轮 {epoch:3d} | loss {loss.item():.4f} | 准确率 {acc.item()*100:.1f}%")


import matplotlib.pyplot  as plt
import numpy as np

# 取出模型参数
w = model.linear.weight.data[0]     # (2,)
b = model.linear.bias.data[0]       # 标量

x_np = x.numpy()
y_np = y.numpy().ravel()

plt.figure(figsize=(6, 6))
plt.scatter(x_np[y_np==0, 0], x_np[y_np==0, 1], label="类 0", alpha=0.6)
plt.scatter(x_np[y_np==1, 0], x_np[y_np==1, 1], label="类 1", alpha=0.6)

# 决策边界: w1*x1 + w2*x2 + b = 0  →  x2 = -(w1*x1 + b)/w2
x_line = np.linspace(-6, 6, 100)
y_line = -(w[0].item() * x_line + b.item()) / w[1].item()
plt.plot(x_line, y_line, 'k--', label="决策边界")

plt.legend()
plt.title("逻辑回归决策边界")
plt.show()