import chainer
import chainer.links as L
import chainer.functions as F
import collections

class TestDNN(chainer.Chain):
    def __init__(self):
        super(TestDNN, self).__init__()
        with self.init_scope():
            self.layer0 = L.Linear(None, 10)
            self.layer1 = L.Linear(10, 10)
            self.layer2 = L.Linear(10, 1)

    def forward(self, x):
        LF = Layers()
        LFF = LF.functions()
        h = x
        for key, funcs in LFF.items():
            for func in funcs:
                h = func(h)
        return h

class Layers(TestDNN):
    def __init__(self):
        super(Layers, self).__init__()
    
    def functions(self):
        return collections.OrderedDict([
            ('layer0', [self.layer0, F.relu]),
            ('layer1', [self.layer1, F.relu]),
            ('layer2', [self.layer2, F.relu]),
        ])