from mainDNN import TestDNN
import numpy as np
from chainer import optimizers, functions

DNN = TestDNN()
opt = optimizers.Adam().setup(DNN)

for i in range(10):
    x = np.random.rand(25)
    y = np.sin(x)
    x[0][0] = 1

    Y = DNN.forward(x)
    loss = functions.squared_difference(Y, y)
    print("loss = {loss}")
    loss.backward()
    opt.update()