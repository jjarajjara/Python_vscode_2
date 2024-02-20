## NAND , OR 게이트 구현

# def AND(x1, x2):
#     x = np.array([x1,x2]) # 입력
#     w = np.array([0.5, 0.5]) # 가중치
#     b = -0.7 # 편향
#     tmp = np.sum(w*x) + b # 가중치와 입력의 곱의 합에 편향을 더한 값
#     if tmp <= 0:
#         return 0
#     else:
#         return 1


# def NAND(x1, x2):
#     x = np.array([x1,x2])
#     w = np.array([-0.5, -0.5])
#     b = 0.7
#     tmp = np.sum(w*x) + b
#     if tmp <= 0:
#         return 0
#     else:
#         return 1

# def OR(x1, x2):
#     x = np.array([x1,x2])
#     w = np.array([0.5, 0.5])
#     b = -0.2
#     tmp = np.sum(w*x) + b 
#     if tmp <= 0:
#         return 0
#     else:
#         return 1


# ## XOR 게이트는 단층 퍼셉트론으로 표현할 수 없다 

# ## 다층 퍼셉트론으로는 표현 가능하다 AND, NAND, OR 게이트를 조합하여 구현해보자 

# def XOR(x1, x2):
#     s1 = NAND(x1,x2)
#     s2 = OR(x1,x2)
#     y = AND(s1,s2)
#     return y  

## 퍼셉트론에서 신경망으로 

## 신경망 -> 입력층 | 은닉층 | 출력층 으로 구성된다

## 편향 : 뉴런이 얼마나 쉽게 활성화 되는지 제어한다
## 가중치 : 각 신호의 영향력을 제어한다     
## 활성화 함수 : 입력 신호의 총합을 출력 신호로 변환하는 함수   
## 계단함수(Step Function) : 임계값을 경계로 출력이 바뀌는 함수

## 시그모이드 함수(sigmaoid function) : 신경망에서 자주 이용하는 활성화 함수 중 하나
## h(x) = 1 / (1 + exp(-x))
## exp(-x)는 e^-x를 뜻함

## 계단함수 구현

# def step_function(x):
#     if x > 0:
#         return 1  
#     else:
#         return 0
    
## 넘파이 배열도 지원하도록 수정
# def step_function(x):
#     y = x > 0
#     return y.astype(np.int) # astype() : 넘파이 배열의 자료형을 변환한다

# x = np.array([-1.0, 1.0, 2.0])
# # print(x)

# y = x > 0
# print(y)

## 넘파이 배열의 자료형을 변환할 때는 astype() 메서드를 이용한다
## 원하는 자료형을 변환할 때 astype(np.int)처럼 np.int를 인수로 지정한다
## (np.변환하고 싶은 인수)

# y = y.astype(np.int)
# print(y)

## 계단 함수의 그래프를 구현해보자 -5.0에서 5.0까지 0.1 간격의 넘파이 배열을 생성
## 책에선 def step_function(x): return np.array(x > 0, dtype=np.int) 로 구현했지만,
## dtype=np.int를 생략해도 잘 작동한다 
## 이유가 뭘까? -> dtype=np.int를 생략하면 np.array()가 자동으로 dtype을 int64로 지정하기 때문이다
## int64는 64비트 정수형을 뜻한다
## 코파일럿 고마워,, 감동이야,,,구글링이 필요가 없네,,,

# def step_function(x):
#     return np.array(x > 0)

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1) # y축의 범위 지정
# plt.show()


## 시그모이드 함수 구현하기 
## 시그모이드 함수란 'S자 모양'이라는 뜻을 가진다

## 넘파이 배열이어도 올바른 결과가 나온다
## 시그모이드 함수 그래프 그리기
# def sigmaoid(x):
#     return 1 / (1 + np.exp(-x))    
 
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmaoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1) ## y축의 범위 지정
# plt.show()
    # x = np.array([-1.0, 1.0, 2.0])
    # sigmaoid(x)

    # t = np.array([1.0, 2.0, 3.0])
    # 1.0 + t 
    # 1.0 / t


## 계단 함수와 시그모이드 함수 공통점 
## = 비선형 함수 

## sigmoid 함수와 계단 함수 동시에 그리기 성공 !!
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def step_function(x):
#     return np.array(x > 0)

# x = np.arange(-5.0, 5.0, 0.1)
# y1 = step_function(x)
# y2 = sigmoid(x)
# plt.plot(x,y1)
# plt.plot(x,y2, 'k--') ## k-- : 검은색 점선
# plt.ylim(-0.1, 1.1)
# plt.show()

## ReLU 함수 
## 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수

# def relu(x):
#     return np.maximum(0,x) ## maximum() : 두 입력 중 큰 값을 선택해 반환하는 함수

## 책에선 주로 ReLU 함수를 사용한다고 한다

## 다차원 배열

# A = np.array([1,2,3,4])
# print(A)
# print(np.ndim(A)) ## np.ndim() 함수 : 배열의 차원 수를 반환하는 함수
# print(A.shape) ## 배열의 형상을 반환하는 함수
# print(A.shape[0])

## shape() 함수는 튜플을 반환한다
## 배열의 형상이란 각 차원의 요소 수를 튜플로 표시한 것
## 튜플은 (1,2)처럼 괄호로 둘러싸인 '쉼표로 구분된 값'의 나열이다
### 다차원 배열인 경우 ) 배열의 갯수, 배열 안의 원소 갯수 
### 1차원 배열인 경우 ) 배열 안의 원소 갯수, (공백)


# B = np.array([[1,2], [3,4], [5,6]])
# print(B)
# print(np.ndim(B))
# print(B.shape)


## 지긋지긋한 행렬의 곱 
# A = np.array([[1,2], [3,4]])
# print(A.shape)
# B = np.array([[5,6], [7,8]])
# print(B.shape)
# print(np.dot(A,B)) ## np.dot() 함수 : 행렬의 곱을 계산하는 함수


# x = np.array([1,2])
# print(x.shape)
# w = np.array([[1,3,5], [2,4,6]])
# print(w)
# print(w.shape)
# y = np.dot(x,w)
# print(y)

# A = np.array([[1,2,3], [4,5,6]])
# print(A.shape)
# B = np.array([[1,2], [3,4], [5,6]])
# print(B.shape)
# print(np.dot(A,B))

# C = np.array([[1,2], [3,4]])
# print(C.shape)
# print(A.shape)
# print(np.dot(A,C)) ## 행렬의 곱에서는 형상에 주의해야 한다

# A = np.array([[1,2], [3,4], [5,6]])
# print(A.shape)
# B = np.array([7,8])
# print(B.shape)
# print(np.dot(A,B)) ## 1차원 배열을 2차원 배열로 변환하여 계산한다

## 신경망에서의 행렬 곱

# x = np.array([1,2])
# print(x.shape)
# w = np.array([[1,3,5], [2,4,6]])
# print(w)
# print(w.shape)
# y = np.dot(x,w)
# print(y)

## 3층 신경망 구현하기

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# x = np.array([1.0, 0.5])
# w1 = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
# b1 = np.array([0.1,0.2,0.3])

# print(w1.shape)
# print(x.shape)
# print(b1.shape)

# a1 = np.dot(x,w1) + b1

# Z1 = sigmoid(a1)
# print(Z1)
# print(a1)

# W2 = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
# b2 = np.array([0.1,0.2])

# print(Z1.shape)
# print(W2.shape)
# print(b2.shape)

# a2 = np.dot(Z1,W2) + b2
# Z2 = sigmoid(a2)

# def identity_function(x):
#     return x

# W3 = np.array([[0.1,0.3], [0.2,0.4]])
# b3 = np.array([0.1,0.2])

# a3 = np.dot(Z2,W3) + b3
# Y = identity_function(a3)


# ## 구현 정리
# ## init_network() : 가중치와 편향을 초기화하고 딕셔너리 변수 network에 저장한다
# ## 딕셔너리 변수 network에는 각 층에 필요한 매개변수(가중치, 편향)을 저장한다

# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
#     network['b1'] = np.array([0.1,0.2,0.3])
#     network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
#     network['b2'] = np.array([0.1,0.2])
#     network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
#     network['b3'] = np.array([0.1,0.2])

#     return network

# ## forward() : 입력 신호를 출력으로 변환하는 처리 과정을 모두 구현한다
# ## 입력 신호가 순방향(입력에서 출력 방향)으로 전달됨에 주의하자

# def forward(network, x):
#     W1, W2, W3 = network['W1'],network['W2'],network['W3']
#     b1, b2, b3 = network['b1'],network['b2'],network['b3']

#     a1 = np.dot(x,W1) + b1
#     Z1 = sigmoid(a1)
#     a2 = np.dot(Z1,W2) + b2
#     Z2 = sigmoid(a2)
#     a3 = np.dot(Z2,W3) + b3
#     Y = identity_function(a3)

#     return Y

# network = init_network()
# x = np.array([1.0,0.5])
# y = forward(network, x)
# print(y)

## 출력층 설계하기
## 일반적으로 회귀에는 항등 함수를, 분류에는 소프트맥스 함수를 사용한다
## 항등 함수 : 입력을 그대로 출력한다
## 소프트맥스 함수 : 입력 신호를 정규화하여 출력한다

## 기계학습 문제는 분류(classification)와 회귀(regression)로 나눌 수 있다
## 분류(classification) : 데이터가 어느 클래스에 속하느냐 문제
## 회귀(regression) : 입력 데이터에서 (연속적인) 수치를 예측하는 문제

## 항등 함수(identity function) : 입력을 그대로 출력하는 함수
## 출력층에서 항등 함수를 사용하면 입력 신호가 그대로 출력 신호가 된다

## 소프트맥스 함수(softmax function) : 입력 값을 정규화하여 출력한다

## yk = exp(ak) / sigma(i=1~n) exp(ai)

## n : 출력층의 뉴런 수
## yk : k번째 출력
## ak : k번째 출력의 입력 신호


## 소프트맥스 함수 구현하기

# a = np.array([0.3,2.9,4.0])

# exp_a = np.exp(a) ## 지수 함수
# print(exp_a)

# sum_exp_a = np.sum(exp_a) ## 지수 함수의 합
# print(sum_exp_a)

# y = exp_a / sum_exp_a
# print(y)

# ## exp_a 함수란? : 밑(base)이 자연상수 e인 지수 함수


# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a

#     return y

# a = np.array([1010,1000,990])
# print(np.exp(a) / np.sum(np.exp(a)))

 ## 소프트맥스 함수의 계산 결과는 모두 0이 되어버린다
## nan : not a number
## 해결책 : 입력 신호 중 최댓값을 빼주면 올바르게 계산할 수 있다

# c = np.max(a)  ## c = 1010
# print(a-c)

# print(np.exp(a-c) / np.sum(np.exp(a-c)))

## 소프트맥스 함수 구현 정리
## 1. 입력 신호 중 최댓값을 빼준다(오버플로 대책)
## 2. exp() 함수를 적용한다
## 3. exp() 함수의 출력을 모두 더한다
## 4. 3의 결과로 나온 값을 분모와 분자로 나눈다

# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a-c) ## 오버플로 대책
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a

#     return y

# ## 소프트맥스 함수의 특징
# ## 소프트맥스 함수의 출력은 0에서 1.0 사이의 실수이다

# a = np.array([0.3,2.9,4.0])
# y = softmax(a)
# print(y)
# print(np.sum(y)) ## 소프트맥스 함수의 출력은 0에서 1.0 사이의 실수이다

## 소프트맥스 함수의 출력 총합은 1이다
##  -> 이 성질 덕분에 소프트맥스 함수의 출력을 '확률'로 해석할 수 있다

## 기계 학습의 문제 풀이 학습과 추론의 두 단계
## 학습 : 모델을 학습하하는 것 -> 훈련 데이터를 사용하여 가중치 매개변수를 학습하는 것
## 추론 : 앞서 학습한 모델로 미지의 데이터에 대해서 추론(분류)하는 것

## 신경망에서는 학습 때는 Softmax 함수를 사용하고, 
## 추론 때는 Softmax 함수를 생략하는 것이 일반적이다

# import sys, os
# sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# from dataset.mnist import load_mnist
# from PIL import Image

# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(flatten=True, normalize=False)

# ## 각 데이터의 형상 출력
# print(x_train.shape) 
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

## load_mnist() 함수는 읽은 MNIST 데이터를 
## (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 형식으로 반환한다

## 인수로는 normalize, flatten, one_hot_label 세 가지를 설정 가능 

## normalize : 입력 이미지의 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화할지 정한다
## false로 설정시 입력 이미지의 픽셀은 원래 값 그대로 0 ~ 255 사이의 값을 유지한다

## flatten : 입력 이미지를 평탄하게, 즉 1차원 배열로 만들지를 정한다
## false로 설정시 입력 이미지를 1 x 28 x 28의 3차원 배열로,
## true로 설정시 784개의 원소로 이루어진 1차원 배열로 저장한다

## one_hot_label : 원-핫 인코딩 형태로 저장할지 정한다
## 원-핫 인코딩이란 정답을 뜻하는 원소만 1이고 나머지는 0인 배열이다
## one_hot_label이 false면 '7'이나 '2'와 같은 레이블을 숫자 그대로 저장한다


# # def img_show(img):
# #     pil_img = Image.fromarray(np.uint8(img)) ## numpy로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
# #     pil_img.show()

# # (x_train, t_train), (x_test, t_test) = \
# #     load_mnist(flatten=True, normalize=False)

# # img = x_train[0]
# # label = t_train[0]
# # print(label) # 5

# # print(img.shape) # (784,)
# # img = img.reshape(28,28) # 원래 이미지의 모양으로 변형
# # print(img.shape) # (28,28)

# # img_show(img)

# ## MNIST 데이터셋을 이용한 신경망의 추론 처리

# ## 신경망의 추론 처리 구성 
# ## 1. 입력층 뉴런 : 784개(이미지 크기 : 28 x 28)
# ## 2. 출력층 뉴런 : 10개(0~9까지의 숫자를 구분)

# ## 입력층 뉴런 784개 -> 은닉층 뉴런 50개 -> 은닉층 뉴런 100개 -> 출력층 뉴런 10개


# from dataset.mnist import load_mnist
# from PIL import Image
# import pickle
# from common.functions import sigmoid, softmax


# def get_data():
#     (x_train, t_train), (x_test, t_test) = \
#         load_mnist(flatten=True, normalize=False, one_hot_label=False)

#     return x_test, t_test

# def init_network():
#     with open("sample_weight.pkl", 'rb') as f: ## 가중치와 편향 매개변수를 sample_weight.pkl에 저장
#         network = pickle.load(f) ## pickle.load() 함수로 파일에서 데이터를 읽는다
#     ## f란? : sample_weight.pkl 파일을 열어서 f에 저장한다
#     return network 

# def predict(network, x): ## 입력 x가 주어졌을 때의 출력 y를 구하는 처리 과정
#     W1, W2, W3 = network['W1'],network['W2'],network['W3']
#     b1, b2, b3 = network['b1'],network['b2'],network['b3']

#     a1 = np.dot(x,W1) + b1
#     Z1 = sigmoid(a1)
#     a2 = np.dot(Z1,W2) + b2
#     Z2 = sigmoid(a2)
#     a3 = np.dot(Z2,W3) + b3
#     y = softmax(a3)

#     return y

## 정확도 평가  

# x, t = get_data()
# network = init_network()

# accuracy_cnt = 0
# for i in range(len(x)): ## len(x) : 10,000
#     y = predict(network, x[i])
#     p = np.argmax(y) ## 확률이 가장 높은 원소의 인덱스를 얻는다
#     if p == t[i]:
#         accuracy_cnt += 1


# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

## 정규화 (normalization) : 데이터를 특정 범위로 변환하는 처리
## 전처리(pre-processing) : 신경망의 입력 데이터에 특정 변환을 가하는 것
## 백색화 (whitening) : 데이터를 균일하게 분포시키는 처리

## 배치처리 

# x, _ = get_data()
# network = init_network()
# W1, W2, W3 = network['W1'],network['W2'],network['W3']
# print(x.shape) ## (10000, 784)
# print(x[0].shape) ## (784,)
# print(W1.shape) ## (784, 50)
# print(W2.shape) ## (50, 100)
# print(W3.shape) ## (100, 10)

## 배치 (batch ) : 하나로 묶은 입력 데이터
## 배치 처리의 이점 
## 1. 수치 계산 라이브러리들이 큰 배열을 효율적으로 처리할 수 있도록 최적화되어 있다
## 2. 커다란 신경망에서는 데이터 전송이 병목으로 작용하는 경우가 자주 있는데,
## 큰 배열을 한꺼번에 계산하는 것이 데이터 전송을 효율적으로 해준다


## 배치 처리 구현
# x, t = get_data()
# network = init_network()

# batch_size = 100 ## 배치 크기
# accuracy_cnt = 0

# for i in range(0, len(x), batch_size): ## range(시작 인덱스, 끝 인덱스, 스텝)
#     x_batch = x[i:i+batch_size] ## x[0:100], x[100:200], x[200:300] ... 
#     y_batch = predict(network, x_batch)
#     p = np.argmax(y_batch, axis=1) ## axis=1 : 1번째 차원을 구성하는 각 원소에서 최댓값을 찾도록 지정
#     accuracy_cnt += np.sum(p == t[i:i+batch_size])

# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))    

## range() 함수 : range(start, end) -> start부터 end-1까지의 숫자를 포함하는 range 객체를 반환한다
## range() 함수 : range(start, end, step) 
## -> start부터 end-1까지의 숫자를 step 간격으로 포함하는 range 객체를 반환한다

## x[i.i+batch_size] : 입력 데이터의 i번째 데이터부터 i+batch_size-1번째 데이터까지 묶는다

## argmax() 함수 : 최댓값의 인덱스를 가져온다
## axis : 최댓값을 구하는 축을 지정한다


## CHAPTER 4 신경망 학습

## 기계학습 : 데이터를 사용해 문제를 해결하는 것
## 훈련 데이터(training data) 와 시험 데이터(test data)로 나누어 학습과 실험을 수행한다

## 범용 능력을 제대로 평가하기 위해 훈련 데이터와 시험 데이터를 분리

## 손실 함수(loss function) : 신경망 성능의 '나쁨'을 나타내는 지표
## 손실 함수의 결과값을 가장 작게 만드는 매개변수 값을 찾는 것이 목표

## 오차제곱합(sum of squares for error, SSE) : 가장 많이 쓰이는 손실 함수

## E = 1/2 sigma(k=1~n) (yk - tk)^2
## yk : 신경망의 출력(신경망이 추정한 값)
## tk : 정답 레이블
## k : 데이터의 차원 수

## 오차제곱합 구현  

# def sum_squares_error(y, t):
#     return 0.5 * np.sum((y-t)**2)

# t = [0,0,1,0,0,0,0,0,0,0] ## 정답은 '2'

# y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0] ## '2'일 확률이 가장 높다고 추정함

# print(sum_squares_error(np.array(y), np.array(t))) ## 0.09750000000000003

# y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0] ## '7'일 확률이 가장 높다고 추정함

# print(sum_squares_error(np.array(y), np.array(t))) ## 0.5975

## 교차 엔트로피 오차(cross entropy error, CEE)

## E = - sigma(k=1~n) tk*logyk
## log : 밑이 e인 자연로그
## yk : 신경망의 출력
## tk : 정답 레이블(원-핫 인코딩)

## 엔트로피 오차 구현 

# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta)) ## delta : 아주 작은 값(0.0000001) -> np.log() 함수에 0을 입력하면 마이너스 무한대를 뜻하는 -inf가 되어 더 이상 계산을 진행할 수 없다

# t = [0,0,1,0,0,0,0,0,0,0] ## 정답은 '2'
# y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0] ## '2'일 확률이 가장 높다고 추정함
# print(cross_entropy_error(np.array(y), np.array(t))) ## 0.510825457099338

# y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0] ## '7'일 확률이 가장 높다고 추정함
# print(cross_entropy_error(np.array(y), np.array(t))) ## 2.302584092994546

## E = -1/N sigma(k=1~n) sigma(n=1~N) tnk*logynk
## N : 데이터의 수  -> 데이터 하나당 오차를 평균낸 것
## tnk : n번째 데이터의 k번째 값을 의미
## ynk : 신경망의 출력
## log : 밑이 e인 자연로그

## 미니배치
## 훈련 데이터로부터 일부만 골라 학습을 수행 

## 미니배치 학습 구현

# import sys, os
# sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np
# from dataset.mnist import load_mnist

# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(normalize=True, one_hot_label=True) 

# ## normalize=True : 입력 이미지의 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화한다
# ## one_hot_label=True : 원-핫 인코딩 형태로 저장한다

# print(x_train.shape) ## (60000, 784)
# print(t_train.shape) ## (60000, 10)

# ## np.random.choice() 함수 : 지정한 범위의 수 중에서 무작위로 원하는 개수만 꺼낼 수 있다

# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size) ## 0~59999 사이의 수 중에서 무작위로 10개를 골라낸다
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

# print(batch_mask)

## 수치 미분의 예 

# import numpy as np
# import matplotlib.pylab as plt

# def numerical_diff(f, x):
#     h = 1e-4
#     return (f(x+h) - f(x-h)) / (2*h) 
# ## 중심 차분(중앙 차분) : x+h와 x-h일 때의 함수 f의 차분을 계산

# def function_1(x):
#     return 0.01*x**2 + 0.1*x ## y = 0.01x^2 + 0.1x

# x = np.arange(0.0, 20.0, 0.1) ## 0에서 20까지 0.1 간격의 배열 x를 만든다
# y = function_1(x)

# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x,y)
# plt.show()

# print(numerical_diff(function_1, 5)) ## 0.1999999999990898
# print(numerical_diff(function_1, 10)) ## 0.2999999999986347

# import numpy as np
# import matplotlib.pylab as plt

# ## 편미분 
# def function_2(x):
#     return x[0]**2 + x[1]**2 ## 또는 np.sum(x**2)

# import sys, os
# sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np

# # ## 기울기 
# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x) ## x와 형상이 같은 배열을 생성

#     for idx in range(x.size):
#         tmp_val = x[idx]
#         ## f(x+h) 계산
#         x[idx] = tmp_val + h
#         fxh1 = f(x)

#         ## f(x-h) 계산
#         x[idx] = tmp_val - h
#         fxh2 = f(x)

#         grad[idx] = (fxh1-fxh2) / (2*h)
#         x[idx] = tmp_val

#     return grad

# # print(numerical_gradient(function_2, np.array([3.0,4.0]))) ## [6. 8.]
# # print(numerical_gradient(function_2, np.array([0.0,2.0]))) ## [0. 4.]
# # print(numerical_gradient(function_2, np.array([3.0,0.0]))) ## [6. 0.]


# ## 경사법 (gradient method) : 기울기를 잘 이용해 함수의 최솟값을 찾으려는 것
# ## 경사 하강법 (gradient descent method) : 최솟값 찾는 방법 ( 자주 쓰임 )
# ## 경사 상승법 (gradient ascent method) : 최댓값 찾는 방법

# def function_2(x):
#     return x[0]**2 + x[1]**2

# def gradient_descent(f, init_x, lr=0.01, step_num=100):
#     x = init_x

#     for i in range(step_num):
#         grad = numerical_gradient(f,x)
#         x -= lr * grad

#     return x

# init_x = np.array([-3.0,4.0]) ## 초기값 설정

# init_x = np.array([-3.0,4.0]) ## 초기값 설정

# print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)) ## 학습률이 너무 작은 예 : 거의 갱신되지 않은 채 끝남
# print(gradient_descent(function_2, init_x=init_x, lr=1e-5, step_num=100)) ## 학습률이 너무 작은 예 : 거의 갱신되지 않은 채 끝남 


# ## 하이퍼파라미터 (hyper parameter) : 학습률 같은 매개변수 


## 신경망에서의 기울기 

# import sys, os
# sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np
# from common.functions import softmax, cross_entropy_error
# from common.gradient import numerical_gradient

# class simpleNet:
#     def __init__(self):
#         self.W = np.random.randn(2,3) # 정규분포로 초기화

#     def predict(self, x):
#         return np.dot(x, self.W)

#     def loss(self, x, t):
#         z = self.predict(x)
#         y = softmax(z)
#         loss = cross_entropy_error(y, t)

#         return loss

# net = simpleNet()
# print(net.W) ## 가중치 매개변수

# x = np.array([0.6, 0.9]) ## 입력 데이터
# p = net.predict(x)
# print(p) ## 예측값
# print(np.argmax(p)) ## 최댓값의 인덱스

# t = np.array([0,0,1]) ## 정답 레이블
# print(net.loss(x,t)) ## 손실 함수의 값


## 학습 알고리즘 구현하기

## 1단계 - 미니배치

## 2단계 - 기울기 산출
    ## 훈련 데이터 중 일부를 무작위로 가져온다
    ## 선별한 데이터를 미니배치라 한다
    ## 미니배치의 손실 함수 값을 줄이는 것이 목표

## 3단계 - 매개변수 갱신
    ## 미니배치의 손실 함수 값을 줄이기 위해 매개변수의 기울기를 구한다 
    ## 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시한다

## 4단계 - 1~3단계 반복
    ## 가중치 매개변수 기울기 방향으로 아주 조금 갱신

## 확률적 경사 하강법 (stochastic gradient descent, SGD)
    ## 확률적으로 무작위로 골라낸 데이터에 대해 수행하는 경사 하강법

## SGD : 대부분의 딥러닝 프레임워크가 확률적 경사 하강법을 구현한 함수를 제공한다


## 2층 신경망 클래스 구현하기 
# import sys, os
# sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np
# from common.functions import *
# from common.gradient import numerical_gradient

# class TwoLayerNet:
#     def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
#         ## 가중치 초기화
#         self.params = {}
#         self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)
    
#     def predict(self, x):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']

#         a1 = np.dot(x,W1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1,W2) + b2
#         y = softmax(a2)

#         return y
    
#     ## x : 입력 데이터, t : 정답 레이블
#     ## predict()의 결과와 정답 레이블을 바탕으로 교차 엔트로피 오차를 구한다
#     def loss(self, x, t):
#         y = self.predict(x)

#         return cross_entropy_error(y, t)
    
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)

#         accuracy = np.sum(y==t) / float(x.shape[0])

#         return accuracy
    
#     ## x : 입력 데이터, t : 정답 레이블
#     ## 수치 미분 방식으로 매개변수의 손실 함수에 대한 기울기 계산 
#     def numercial_gradient(self, x, t):
#         loss_W = lambda W: self.loss(x, t)

#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

#         return grads
    
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

 

# import numpy as np
# from dataset.mnist import load_mnist
# from two_layer_net import TwoLayerNet

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# train_loss_list = []

# ## 하이퍼파라미터
# iters_num = 10000 ## 반복 횟수
# train_size = x_train.shape[0]
# batch_size = 100 ## 미니배치 크기
# learning_rate = 0.1

# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# for i in range(iters_num):
#     ## 미니배치 획득
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]

#     ## 기울기 계산
#     grad = network.numercial_gradient(x_batch, t_batch)
#     ## grad = network.gradient(x_batch, t_batch) ## 성능 개선판

#     ## 매개변수 갱신
#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * grad[key]

#     ## 학습 경과 기록
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss)

#     print(loss)

## 결과값을 보면 학습 횟수가 늘어가면서 손실 함수의 값이 줄어든다 
## 신경망의 가중치 매개변수가 서서히 데이터에 적응하고 있음을 의미한다 
## 데이터를 반복해서 학습함으로 최적 가중치 매개변수로 서서히 다가가고 있다
    

## 시험 데이터로 평가하기 
## 손실 함수의 값 : 훈련 데이터의 미니배치에 대한 손실 함수의 값

## 에폭(epoch) : 하나의 단위로, 학습에서 훈련 데이터를 모두 소진했을 때의 횟수를 의미한다

# import numpy as np
# from dataset.mnist import load_mnist
# from two_layer_net import TwoLayerNet

# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(normalize=True, one_hot_label=True)
    
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# iters_num = 10000
# train_size = x_train.shape[0]
# batch_size = 100
# learning_rate = 0.1

# train_loss_list = []
# train_acc_list = []
# test_acc_list = []

# ## 1에폭당 반복 수
# iter_per_epoch = max(train_size / batch_size, 1)

# for i in range(iters_num):
#     ## 미니배치 획득
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]

#     ## 기울기 계산
#     grad = network.numercial_gradient(x_batch, t_batch)
#     ## grad = network.gradient(x_batch, t_batch) ## 성능 개선판

#     ## 매개변수 갱신
#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * grad[key]

#     ## 학습 경과 기록
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss)

#     ## 1에폭당 정확도 계산
#     if i % iter_per_epoch == 0:
#         train_acc = network.accuracy(x_train, t_train)
#         test_acc = network.accuracy(x_test, t_test)
#         train_acc_list.append(train_acc)
#         test_acc_list.append(test_acc)
#         print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


## 오차역전파법 ( 가중치 매개변수의 기울기를 효율적으로 계산 )

## 1. 수식을 통한 것 ( 일반적인 방법 )
## 2. 계산 그래프를 통한 것 

## 계산 그래프 (computational graph) : 계산 과정을 그래프로 나타낸 것

## 노드(node) : 원(노드)로 표현
## 에지(edge) : 노드 사이의 연결선을 의미

## 단순한 계층 구현
 

## 곱셈 계층    
# class MulLayer:
#     def __init__(self):
#         self.x = None
#         self.y = None

#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#         out = x * y

#         return out

#     def backward(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x

#         return dx, dy
    
# ## 덧셈 계층 
# class AddLayer:
#     def __init__(self):
#         pass

#     def forward(self, x, y):
#         out = x + y
#         return out

#     def backward(self, dout):
#         dx = dout * 1
#         dy = dout * 1

#         return dx, dy
    
## 활성화 함수 계층 구현하기 

# import numpy as np

# class Relu: 
#     def __init__(self):
#         self.mask = None
    
#     def forward(self, x):
#         self.mask = (x <= 0)
#         out = x.copy()
#         out[self.mask] = 0

#         return out

#     def backword(self, dout):
#         dout[self.mask] = 0
#         dx = dout

#         return dx   
    
# x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# print(x)

# mask = (x <= 0)
# print(mask)

# class Sigmoid:
#     def __init__(self):
#         self.out = None

#     def forward(self, x):
#         out = 1 / (1 + np.exp(-x))
#         self.out = out

#         return out

#     def backward(self, dout):
#         dx = dout * (1.0 - self.out) * self.out

#         return dx
    

## Affine/Softmax 계층 구현하기

# import numpy as np  

# X = np.random.rand(2) ## 입력
# W = np.random.rand(2,3) ## 가중치
# B = np.random.rand(3) ## 편향

# X.shape ## (2,)
# W.shape ## (2,3)
# B.shape ## (3,)

# Y = np.dot(X,W) + B 

# ## 어파인 변환 (Affine transformation) : 신경망의 순전파 처리 과정을 수식으로 나타낸 것

# X_dot_W = np.array([[0,0,0],[10,10,10]])
# B = np.array([1,2,3])

# print(X_dot_W)
# print(X_dot_W + B)

# dY = np.array([[1,2,3],[4,5,6]])
# print(dY)

# dB = np.sum(dY, axis=0)
# print(dB)


## 오차역전파법 구현하기

## 신경망 학습의 전체 그림

## 전제 : 신경망에는 적응 가능한 가중치와 편향이 있고, 
## 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습'이라 한다

## 1단계 - 미니배치
## 훈련 데이터 중 일부를 무작위로 가져온다. 이렇게 선별한 데이터를 미니배치라 하고,
## 그 미니배치의 손실 함수 값을 줄이는 것이 목표이다

## 2단계 - 기울기 산출
## 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다.

## 3단계 - 매개변수 갱신
## 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다

## 4단계 - 반복

## 오차역전파법을 적용한 신경망 구현하기

# import sys, os 
# sys.path.append(os.pardir)
# import numpy as np
# from common.layers import *
# from common.gradient import numerical_gradient
# from collections import OrderedDict

# class TwoLayerNet:

#     def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
#         ## 가중치 초기화
#         self.params = {}
#         self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)

#         ## 계층 생성
#         self.layers = OrderedDict()
#         self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
#         self.layers['Relu1'] = Relu()
#         self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

#         self.lastLayer = SoftmaxWithLoss()

#     def predict(self, x):
#         for layer in self.layers.values():
#             x = layer.forward(x)

#         return x
    
#     ## x : 입력 데이터, t : 정답 레이블
#     def loss(self, x, t):
#         y = self.predict(x)
#         return self.lastLayer.forward(y, t)
    
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         if t.ndim != 1 : t = np.argmax(t, axis=1)

#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy
    
#     ## x : 입력 데이터, t : 정답 레이블
#     def numerical_gradient(self, x, t):
#         loss_W = lambda W: self.loss(x, t)

#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

#         return grads
    
#     def gradient(self, x, t):
#         ## 순전파
#         self.loss(x, t)

#         ## 역전파
#         dout = 1
#         dout = self.lastLayer.backward(dout)

#         layers = list(self.layers.values())
#         layers.reverse()
#         for layer in layers:
#             dout = layer.backward(dout)

#         ## 결과 저장
#         grads = {}
#         grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
#         grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

#         return grads
    
## gradient check ( 기울기 확인 )
    
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


## 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

## 각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 낸다.
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
