
## 인프설 그래프 그리기 

from itertools import chain
import numpy as np 
from numpy import ndarray
##ndarray : 다차원 배열을 나타내는 자료형 
import matplotlib.pyplot as plt
from typing import Callable
from typing import List

np.set_printoptions(precision=4) ## 소수점 4자리까지 출력

def square(x: ndarray) -> ndarray:
    ## ndarray를 입력받아 ndarray를 출력
    return np.power(x,2) ## x의 제곱을 반환

def relu(x: ndarray) -> ndarray: 
    return np.maximum(x,0) ## x와 0중 큰 값을 반환

def deriv(func: Callable[[np.ndarray], np.ndarray], input_: np.ndarray, delta:float = 0.0001) -> np.ndarray: 
    ## 함수, 입력, 델타를 입력받아 ndarray를 출력
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta) 
    ## 미분을 반환

Array_Function = Callable[[np.ndarray], np.ndarray] 
Chain = List[Array_Function] ## Array_Function의 리스트


def chain_length_2(chain: Chain, x: np.ndarray) -> np.ndarray: ## 체인과 ndarray를 입력받아 ndarray를 출력
    assert len(chain) == 2 ## 체인의 길이가 2인지 확인
    f1 = chain[0] 
    f2 = chain[1]

    return f2(f1(x)) 

def chain_deriv_2(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    assert len(chain) == 2
    assert input_range.ndim == 1 ## 입력이 1차원인지 확인

    f1 = chain[0]
    f2 = chain[1]

    f1_of_x = f1(input_range) ## f1의 결과값
    df1dx = deriv(f1, input_range) 
    df2du = deriv(f2, f1(input_range)) 

    return df1dx * df2du

def sigmoid(x : np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def plot_chain(ax, chain: Chain, input_range: np.ndarray) -> None:
    assert input_range.ndim == 1 ## 입력이 1차원인지 확인
    output_range = chain_length_2(chain, input_range) ## 체인의 결과값
    ax.plot(input_range, output_range) ## 그래프를 그린다
    
def plot_chain_deriv(ax, chain: Chain, input_range: np.ndarray) -> None:
    assert input_range.ndim == 1
    ax.plot(input_range, chain_deriv_2(chain, input_range))

fig, ax = plt.subplots(1,2,sharey=True, figsize=(16,8))

chain_1 = [square, sigmoid]
chain_2 = [sigmoid, square]

PLOT_RANGE = np.arange(-3,3,0.01)
plot_chain(ax[0], chain_1, PLOT_RANGE)
plot_chain_deriv(ax[0], chain_1, PLOT_RANGE)

ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
## $ : 수학식을 입력할 때 사용
## \frac : 분수를 입력할 때 사용
## df : 미분을 나타낸다
## legend : 범례를 나타낸다
ax[0].set_title("Function and derivative for\n$f(x) = sigmoid(square(x))$")
##set_title : 그래프의 제목을 나타낸다

plot_chain(ax[1], chain_2, PLOT_RANGE)
plot_chain_deriv(ax[1], chain_2, PLOT_RANGE)
ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[1].set_title("Function and derivative for\n$f(x) = square(sigmoid(x))")

plt.show()



