from mainDNN import TestDNN
import numpy as np
from chainer import optimizers, functions

def sinDatas(epoch, batch=10):
    out = []
    for i in range(epoch*batch):
        if i==0:
            out.append(np.random.rand(25))
        elif(i%10==0):
            yield np.array(out, dtype=np.float32)
            out = []
        else:
            out.append(np.random.rand(25))

if __name__ == '__main__':
    DNN = TestDNN()
    opt = optimizers.Adam().setup(DNN)
    epoch = 100
    datas = sinDatas(epoch)

    for i in datas:
        x = i
        y = np.sin(x)

        Y = DNN.forward(x)
        loss = np.sum(functions.squared_difference(Y, y))
        print("loss = {}".format(loss))
        loss.backward()
        opt.update()