
import numpy as np
from common.functions import softmax, cross_entropy_error

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):  
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
    
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
    


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx  
    
class OrderedDict:
    def __init__(self):
        self.data = {}
        
    def __setitem__(self, key, value):
        self.data[key] = value
        
    def __getitem__(self, key):
        return self.data[key]
    
    def items(self):
        return self.data.items()
    
    def values(self):
        return self.data.values()
    
    def keys(self):
        return self.data.keys()
    
    def popitem(self):
        return self.data.popitem()
    
    def __len__(self):
        return len(self.data)
    
    def __contains__(self, key):
        return key in self.data
    
    def clear(self):
        self.data.clear()
    
    def __repr__(self):
        return repr(self.data)
    
    def __str__(self):
        return str(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def __reversed__(self):
        return reversed(self.data)
    
    def __eq__(self, value):
        return self.data == value
    
    def __ne__(self, value):
        return self.data != value
    
    def __lt__(self, value):
        return self.data < value
    
    def __le__(self, value):
        return self.data <= value
    
    def __gt__(self, value):
        return self.data > value
    
    def __ge__(self, value):
        return self.data >= value
    
    def __add__(self, value):
        return self.data + value
    
    def __sub__(self, value):
        return self.data - value
    
    def __mul__(self, value):
        return self.data * value
    
    def __truediv__(self, value):
        return self.data / value
    
    def __floordiv__(self, value):
        return self.data // value
    
    def __mod__(self, value):
        return self.data % value
    
    def __divmod__(self, value):
        return divmod(self.data, value)
    
    def __pow__(self, value):
        return self.data ** value
    
    def __lshift__(self, value):
        return self.data << value
    
    def __rshift__(self, value):
        return self.data >> value
    
    def __and__(self, value):
        return self.data & value
    
    def __xor__(self, value):
        return self.data ^ value
    
    def __or__(self, value):
        return self.data | value
    
    