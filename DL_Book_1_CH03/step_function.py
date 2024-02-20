## 퍼셉트론에서 신경망으로 

## 신경망 -> 입력층 | 은닉층 | 출력층 으로 구성된다

## 편향 : 뉴런이 얼마나 쉽게 활성화 되는지 제어한다
## 가중치 : 각 신호의 영향력을 제어한다     
## 활성화 함수 : 입력 신호의 총합을 출력 신호로 변환하는 함수   
## 계단함수(Step Function) : 임계값을 경계로 출력이 바뀌는 함수

import numpy as np
import matplotlib.pyplot as plt

# def step_function(x):
#     y = x > 0
#     return y.astype(np.int) # astype() : 넘파이 배열의 자료형을 변환한다

# x = np.array([-1.0, 1.0, 2.0])
# # print(x)

# y = x > 0
# print(y)

# y = y.astype(np.int)
# print(y)

## 넘파이 배열의 자료형을 변환할 때는 astype() 메서드를 이용한다
## 원하는 자료형을 변환할 때 astype(np.int)처럼 np.int를 인수로 지정한다
## (np.변환하고 싶은 인수)

## 계단 함수의 그래프를 구현해보자 -5.0에서 5.0까지 0.1 간격의 넘파이 배열을 생성
## 책에선 def step_function(x): return np.array(x > 0, dtype=np.int) 로 구현했지만,
## dtype=np.int를 생략해도 잘 작동한다 
## 이유가 뭘까? -> dtype=np.int를 생략하면 np.array()가 자동으로 dtype을 int64로 지정하기 때문이다
## int64는 64비트 정수형을 뜻한다

def step_function(x):
    return np.array(x > 0)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()
