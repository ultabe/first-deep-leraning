import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

x = torch.linspace(-1, 1, 100).unsqueeze(1)
y = 4.0 * x + 2.0 + torch.randn(100, 1) * 0.3

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x)                 
    loss = loss_fn(y_pred, y)          

    optimizer.zero_grad()             
    loss.backward()                    
    optimizer.step()                  

    if epoch % 10 == 0:
        print(f"轮 {epoch:3d} | loss {loss.item():.4f}")

print("学到的 w, b:")
for name, p in model.named_parameters():
    print(name, p.data)