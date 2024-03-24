
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import ndarray

from typing import Callable
from typing import Dict

def square(x: ndarray) -> ndarray:
    '''
    인자로 받은 ndarray 배열의 각 요솟값을 제곱한다
    '''
    return np.power(x,2)

def leaky_relu(x: ndarray) -> ndarray:
    '''
    ndarray 배열의 각 요소에 'Leaky ReLU' 함수를 적용한 결과를 반환한다
    '''
    return np.maximum(0.2*x,x)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6))  
# 2 Rows, 1 Col

input_range = np.arange(-2, 2, 0.01)
ax[0].plot(input_range, square(input_range))
ax[0].plot(input_range, square(input_range))
ax[0].set_title('Square 함수')
ax[0].set_xlabel('입력')
ax[0].set_ylabel('출력')

ax[1].plot(input_range, leaky_relu(input_range))
ax[1].plot(input_range, leaky_relu(input_range))
ax[1].set_title('ReLU 함수')
ax[1].set_xlabel('입력')
ax[1].set_ylabel('출력')

plt.show()
