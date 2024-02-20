
## sigmoid function & step function 동시에 그리기

## 계단 함수와 시그모이드 함수 공통점 
## = 비선형 함수 

import numpy as np  
import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.array(x > 0)

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x,y1)
plt.plot(x,y2, 'k--') ## k-- : 검은색 점선
plt.ylim(-0.1, 1.1)
plt.show()


