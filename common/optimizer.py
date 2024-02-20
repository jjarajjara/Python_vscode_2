import numpy as np

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

## v : 물체의 속도, momentum : 모멘텀 상수, lr : 학습률


## AdaGrad : 과거의 기울기를 제곱하여 계속 더해간다 
## -> 학습을 진행할수록 갱신 강도가 약해진다
## -> 무한히 계속 학습한다면 어느 순간 갱신량이 0이 되어 전혀 갱신되지 않게 된다
## -> 이를 개선한 것이 RMSprop
## -> RMSprop : 먼 과거의 기울기는 서서히 잊고 새로운 정보를 크게 반영 
## 지수이동평균(Exponential Moving Average)
## -> 과거의 기울기의 반영 규모를 기하급수적으로 감소시킨다
            
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            ## 1e-7 : self.h[key]에 0이 담겨있어 0으로 나누는 사태를 막기 위해 작은 값을 더한다

