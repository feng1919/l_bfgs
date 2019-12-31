
import numpy as np
import keras.backend as K

class limited_memory():
    def __init__(self, m, n, dtype):
        self.lm = []
        for i in range(m):
            self.lm[i] = iteration_data(n, dtype)

class iteration_data():
    def __init__(self, n, dtype):
        self.alpha = 0
        self.ys = 0
        self.y = K.placeholder(shape=(n,), dtype=dtype)
        self.s = K.placeholder(shape=(n,), dtype=dtype)
