## ReLU 함수 
## 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x) ## maximum() : 두 입력 중 큰 값을 선택해 반환하는 함수

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()

## 책에선 주로 ReLU 함수를 사용한다고 한다
