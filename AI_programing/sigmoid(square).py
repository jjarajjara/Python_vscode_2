
## 인프설 그래프 그리기 

from itertools import chain
import numpy as np 
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Callable
from typing import List

np.set_printoptions(precision=4)

def square(x: ndarray) -> ndarray:
    return np.power(x,2)

def relu(x: ndarray) -> ndarray:
    return np.maximum(x,0)

def deriv(func: Callable[[np.ndarray], np.ndarray], input_: np.ndarray, delta:float = 0.0001) -> np.ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

Array_Function = Callable[[np.ndarray], np.ndarray]
Chain = List[Array_Function]


def chain_length_2(chain: Chain, x: np.ndarray) -> np.ndarray:
    assert len(chain) == 2
    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))

def chain_deriv_2(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    assert len(chain) == 2
    assert input_range.ndim == 1

    f1 = chain[0]
    f2 = chain[1]

    f1_of_x = f1(input_range)
    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1(input_range))

    return df1dx * df2du

def sigmoid(x : np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def plot_chain(ax, chain: Chain, input_range: np.ndarray) -> None:
    assert input_range.ndim == 1
    output_range = chain_length_2(chain, input_range)
    ax.plot(input_range, output_range)
    
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




