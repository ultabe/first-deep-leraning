import torch
torch.manual_seed(42)
x = torch.linspace(-1, 1, 100).unsqueeze(1)
y = 4.0 * x + 2.0 + torch.randn(100, 1) * 0.3
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
y_pred = x * w + b
loss = ((y_pred - y) ** 2).mean()
lr = 0.1
for epoch in range(100):
    y_pred = x * w + b                    # ① 预测
    loss = ((y_pred - y) ** 2).mean()     # ② 误差
    loss.backward()                       # ③ 算梯度
    with torch.no_grad():                 # ④ 更新参数
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()                        # ⑤ 清零
    b.grad.zero_()
if epoch % 10 == 0:                   # ⑥ 打印
        print(f"轮 {epoch:3d} | loss {loss.item():.4f} | w {w.item():.3f} | b {b.item():.3f}")
print("最终:", w.item(), b.item())