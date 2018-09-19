
import torch.nn as nn

class _Decay(nn.Module):
    def __init__(self, base, _iter, lambda_min):
        super(_Decay, self).__init__()
        self.base = base
        self._iter = _iter
        self.lambda_min = lambda_min

class ExponentialDecay(_Decay):
    def __init__(self, base, _iter, lambda_min, gamma, power):
        super(ExponentialDecay, self).__init__(base, _iter, lambda_min)
        self.gamma = gamma
        self.power = power

    def next(self):
        self._iter += 1
        _lambda = self.base * pow((1 + self.gamma*self._iter), -self.power) 
        _lambda = max(_lambda, self.lambda_min)
        return _lambda

class LinearDecay(_Decay):
    def __init__(self, base, _iter, lambda_min, scale):
        super(LinearDecay, self).__init__(base, _iter, lambda_min)
        self.scale = scale

    def next(self):
        self._iter += 1
        _lambda = self.base * pow(scale, self._iter)
        _lambda = max(_lambda, self.lambda_min)
        return _lambda
        
        
        
