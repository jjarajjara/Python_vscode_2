import numpy as np 
import matplotlib.pyplot as plt
from numpy import ndarray

from typing import Callable
from typing import List

# ndarray를 인자로 받고 ndarray를 반환하는 함수
Array_Function = Callable[[ndarray], ndarray]

# Chain은 함수의 리스트다.
Chain = List[Array_Function]

## 순방향 계산
def matmul_forward(X: ndarray,
                   W: ndarray) -> ndarray:
    '''
    순방향 계산을 행렬곱으로 계산 
    '''
    
    assert X.shape[1] == W.shape[0], \
    '''
    행렬곱을 계산하려면 첫 번째 배열의 열의 개수와
    두 번째 배열의 행의 개수가 일치해야 한다.
    그러나 지금은 첫 번째 배열의 열의 개수가 {0}이고
    두 번째 배열의 행의 개수가 {1}이다.
    '''.format(X.shape[1], W.shape[0])

    # 행렬곱 연산
    N = np.dot(X, W)

    return N

## 역방향 계산
def matmul_backward_first(X: ndarray,
                          W: ndarray) -> ndarray:
    '''
    첫 번째 인자에 대한 행렬곱의 역방향 계산 수행
    '''

    # 역방향 계산
    dNdX = np.transpose(W, (1, 0))

    return dNdX

np.random.seed(190203)

X = np.random.randn(1,3)
W = np.random.randn(3,1)

print(X)
matmul_backward_first(X, W)

## 행렬곱의 순방향 계산
def matrix_forward_extra(X: ndarray,
                         W: ndarray,
                         sigma: Array_Function) -> ndarray:
    '''
    행렬곱이 포함된 함수와 또 다른 함수의 합성함수에 대한 순방향 계산을 수행
    '''
    assert X.shape[1] == W.shape[0]

    # 행렬곱
    N = np.dot(X, W)

    # 행렬곱의 출력을 함수 sigma의 입력값으로 전달
    S = sigma(N)

    return S

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          diff: float = 0.001) -> ndarray:
    '''
    배열 input의 각 요소에 대해 함수 func의 도함숫값 계산
    '''
    return (func(input_ + diff) - func(input_ - diff)) / (2 * diff)

def sigmoid(x: ndarray) -> ndarray:
    '''
    입력으로 받은 ndarray의 각 요소에 대한 sigmoid 함숫값을 계산한다.
    '''
    return 1 / (1 + np.exp(-x))

## 행렬곱의 역방향 계산
def matrix_function_backward_1(X: ndarray,
                               W: ndarray,
                               sigma: Array_Function) -> ndarray:
    '''
    첫 번째 요소에 대한 행렬함수의 도함수 계산
    '''
    assert X.shape[1] == W.shape[0]

    # 행렬곱
    N = np.dot(X, W)

    # 행렬곱의 출력을 함수 sigma의 입력값으로 전달
    S = sigma(N)

    # 역방향 계산
    dSdN = deriv(sigma, N)

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # 계산한 값을 모두 곱함. 여기서는 dNdX의 모양이 1*1이므로 순서는 무관함
    return np.dot(dSdN, dNdX)

print(matrix_function_backward_1(X, W, sigmoid))

def forward_test(ind1, ind2, inc):
    
    X1 = X.copy()
    X1[ind1, ind2] = X[ind1, ind2] + inc

    return matrix_forward_extra(X1, W, sigmoid)

(np.round(forward_test(0, 2, 0.01) - forward_test(0, 2, 0), 4)) / 0.01
print(np.round(forward_test(0, 2, 0.01) - forward_test(0, 2, 0), 4) / 0.01)

np.round(matrix_function_backward_1(X, W, sigmoid)[0, 2], 2)
print(np.round(matrix_function_backward_1(X, W, sigmoid)[0, 2], 2))

