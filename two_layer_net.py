## 2층 신경망 클래스 구현하기 
import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from common.layers import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        ## 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        ## 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    ## x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    ## x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        ## 순전파
        self.loss(x, t)

        ## 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        ## 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    

    # def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    #     ## 가중치 초기화
    #     self.params = {}
    #     self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    #     self.params['b1'] = np.zeros(hidden_size)
    #     self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    #     self.params['b2'] = np.zeros(output_size)
    
    # def predict(self, x):
    #     W1, W2 = self.params['W1'], self.params['W2']
    #     b1, b2 = self.params['b1'], self.params['b2']

    #     a1 = np.dot(x,W1) + b1
    #     z1 = sigmoid(a1)
    #     a2 = np.dot(z1,W2) + b2
    #     y = softmax(a2)

    #     return y
    
    # ## x : 입력 데이터, t : 정답 레이블
    # ## predict()의 결과와 정답 레이블을 바탕으로 교차 엔트로피 오차를 구한다
    # def loss(self, x, t):
    #     y = self.predict(x)

    #     return cross_entropy_error(y, t)
    
    # def accuracy(self, x, t):
    #     y = self.predict(x)
    #     y = np.argmax(y, axis=1)
    #     t = np.argmax(t, axis=1)

    #     accuracy = np.sum(y==t) / float(x.shape[0])

    #     return accuracy
    
    # ## x : 입력 데이터, t : 정답 레이블
    # ## 수치 미분 방식으로 매개변수의 손실 함수에 대한 기울기 계산 
    # def numercial_gradient(self, x, t):
    #     loss_W = lambda W: self.loss(x, t)

    #     grads = {}
    #     grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    #     grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    #     grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    #     grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    #     return grads
    

## TwoLayerNet 클래스의 변수 
    
## params 변수 : 신경망의 매개변수를 보관하는 딕셔너리 변수
    ## params['W1'] : 1번째 층의 가중치, params['b1'] : 1번째 층의 편향
    ## params['W2'] : 2번째 층의 가중치, params['b2'] : 2번째 층의 편향

## grads 변수 : 기울기를 보관하는 딕셔너리 변수(numerical_gradient() 메서드의 반환 값)
    ## grads['W1'] : 1번째 층의 가중치의 기울기, grads['b1'] : 1번째 층의 편향의 기울기
    ## grads['W2'] : 2번째 층의 가중치의 기울기, grads['b2'] : 2번째 층의 편향의 기울기

## TwoLayerNet 클래스의 메서드

## __init__(self, input_size, hidden_size, output_size) : 초기화를 수행한다
    ## input_size : 입력층의 뉴런 수
    ## hidden_size : 은닉층의 뉴런 수
    ## output_size : 출력층의 뉴런 수

## predict(self, x) : 예측(추론)을 수행한다

## loss(self, x, t) : 손실 함수의 값을 구한다

## accuracy(self, x, t) : 정확도를 구한다

## numerical_gradient(self, x, t) : 가중치 매개변수의 기울기를 구한다

## gradient(self, x, t) : 가중치 매개변수의 기울기를 구한다

 