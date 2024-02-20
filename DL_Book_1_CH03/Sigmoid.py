## 시그모이드 함수(sigmaoid function) : 신경망에서 자주 이용하는 활성화 함수 중 하나
## h(x) = 1 / (1 + exp(-x))
## exp(-x)는 e^-x를 뜻함

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()

## 시그모이드 함수는 입력이 커지면 1에 수렴하고, 입력이 작아지면 0에 수렴한다
## 시그모이드 함수는 S자 모양이다
## 시그모이드 함수는 입력이 0일 때 0.5의 값을 출력한다

