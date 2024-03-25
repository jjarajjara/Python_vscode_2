import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd 

from typing import Callable
from typing import Dict 

data_url ="http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
## raw : 날 것의 데이터 , :: : 전체 데이터 중에서 2칸씩 건너뛰어라
target = raw_df.values[1::2, 2] ## 2번째 열에 있는 데이터를 2칸씩 건너뛰어서 가져와라

# print("파이썬 리스트를 사용한 연산 : ")
# a = [1,2,3]
# b = [4,5,6]
# print("a+b", a+b )
# try:
#     print(a*b)
# except TypeError:
#     print("a*b는 파이썬 리스트에서 지원하지 않습니다.")
# print()

# print("넘파이 배열을 사용한 연산 : ")
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# print("a+b", a+b )
# print("a*b", a*b)

# a = np.array([[1,2,3],
#              [4,5,6]])
# print(a)

# b = np.array([10,20,30])
# print("a+b:\n", a+b)

a = np.array([[1,2],
              [3,4]])
print('a:')
print(a)
print('a.sum(axis=0):', a.sum(axis=0)) ## 열의 합
print('a.sum(axis=1):', a.sum(axis=1)) ## 행의 합