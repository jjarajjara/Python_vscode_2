import numpy as np

# x = np.array([1,2,3])
# print(x.__class__)
# print(x.shape)
# print(x.ndim)

# W = np.array([[1,2,3],[4,5,6]])
# # print(W.shape)
# # print(W.ndim)

# X = np.array([[0,1,2],[3,4,5]])

# print(W+X)
# print(W*X)

# A = np.array([[1,2],[3,4]])
# print(A*10)

# b = np.array([10,20])
# print(A*b)  # 브로드캐스트 기능

## np.dot() : 벡터의 내적 
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# print(np.dot(a,b))

# ## np.matmul() : 행렬의 곱
# A = np.array([[1,2],[3,4]])
# B = np.array([[5,6],[7,8]])
# print(np.matmul(A,B))

W1 = np.random.randn(2,4) ## 가중치
b1 = np.random.randn(4) ## 편향
x = np.random.randn(10,2) ## 입력
h = np.matmul(x,W1) + b1 
## 편향의 덧셈은 브로드캐스트 기능으로 가능
## b1의 형상이 (4,)이지만, (10,4)로 확장되어 더해짐

print(h)


