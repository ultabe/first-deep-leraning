import torch

# 1. 创建 3×3
A = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
print(A)
# tensor([[1., 2., 3.],
#         [4., 5., 6.],
#         [7., 8., 9.]])

# 2. 转置
A_T = A.T                      # 或 A.transpose(0, 1)
print(A_T.shape)               # torch.Size([3, 3])
print(torch.equal(A_T.T, A))   # True —— 转置两次回到原样

# 3. 矩阵乘法
print(A @ A_T)                 # 对称矩阵
print(A_T @ A)                 # 也对称，但 ≠ A @ A_T

# 4. 形状变换
B = A.reshape(1, 9)            # 1×9
print(B.shape)
print(B.view(9).shape)         # (9,)
print(B.squeeze().shape)       # (9,)
print(B.squeeze().unsqueeze(1).shape)  # (9, 1)
