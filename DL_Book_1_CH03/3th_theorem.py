# ## 구현 정리
# ## init_network() : 가중치와 편향을 초기화하고 딕셔너리 변수 network에 저장한다
# ## 딕셔너리 변수 network에는 각 층에 필요한 매개변수(가중치, 편향)을 저장한다

import numpy as np
from common.functions import sigmoid, identity_function


def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

# ## forward() : 입력 신호를 출력으로 변환하는 처리 과정을 모두 구현한다
# ## 입력 신호가 순방향(입력에서 출력 방향)으로 전달됨에 주의하자

def forward(network, x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    Z1 = sigmoid(a1)
    a2 = np.dot(Z1,W2) + b2
    Z2 = sigmoid(a2)
    a3 = np.dot(Z2,W3) + b3
    Y = identity_function(a3)

    return Y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)