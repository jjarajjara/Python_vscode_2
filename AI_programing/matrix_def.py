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