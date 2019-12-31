
import keras.backend as K

def vec2norm(x):
    return K.sqrt(K.sum(K.square(x)))

def veccpy(dst, src, n):
    for i in range(n):
        dst[i] = src[i]

def vecncpy(dst, src, n):
    for i in range(n):
        dst[i] = -src[i]

def vecadd(dst, src, c, n):
    for i in range(n):
        dst[i] += src[i] * c

def vecdiff(dst, x, y, n):
    for i in range(n):
        dst[i] = x[i] - y[i]

def vecscale(dst, c, n):
    for i in range(n):
        dst[i] *= c

def vecmul(dst, src, n):
    for i in range(n):
        dst[i] *= src[i]
