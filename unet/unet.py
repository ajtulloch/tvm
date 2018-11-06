import mxnet as mx
import numpy as np


def conv(x, num_filter):
    return mx.sym.Convolution(x, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)

def pool(x):
    return mx.sym.Pooling(x, kernel=(2, 2), stride=(2,2), pad=(0, 0), pool_type='max')

def resize(x):
    return mx.sym.UpSampling(x, scale=2, sample_type="nearest")

def relu(x):
    return mx.sym.relu(x)


def sigmoid(x):
    return mx.sym.sigmoid(x)

def conv_pool(x, f):
    c = conv(x, f)
    p = pool(c)
    r = relu(p)
    return (r, c)

def conv_relu(x, f):
    c = conv(x, f)
    r = relu(c)
    return r

def conv_resize(x, f):
    c = conv(x, f)
    r = resize(c)
    return r

def unet(alignment=None):
    data = mx.sym.Variable(name='data')

    def align(x):
        return x if not alignment else ((x + alignment - 1) // alignment) * alignment

    (r0, p0) = conv_pool(data, align(12))  # 96x96
    (r1, p1) = conv_pool(r0, align(24))  # 48x48
    (r2, p2) = conv_pool(r1, align(48))  # 24x24
    (r3, p3) = conv_pool(r2, align(96))  # 12x12
    (r4, p4) = conv_pool(r3, align(180))  # 6x6
    r5 = conv_relu(r4, align(220))  # 6x6
    r6 = relu(conv_resize(r5, align(180)) + p4)
    r7 = relu(conv_resize(r6, align(96)) + p3)
    r8 = relu(conv_resize(r7, align(48)) + p2)
    r9 = relu(conv_resize(r8, align(24)) + p1)
    r10 = relu(conv_resize(r9, align(12)) + p0)
    r11 = conv(r10, 1)
    s = sigmoid(r11)
    return conv(s, 1)


def down_block():
    data = mx.sym.Variable(name='data')
    return relu(pool(data))

def up_block():
    data = mx.sym.Variable(name='data')
    return relu(resize(data))
