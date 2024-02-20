
import numpy as np

from DL_Book_1_CH02.OR_Gate import OR
from DL_Book_1_CH02.AND_Gate import AND
from DL_Book_1_CH02.NAND_Gate import NAND

def XOR(x1, x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y  

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))

