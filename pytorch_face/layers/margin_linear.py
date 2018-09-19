import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class _MarginLinear(nn.Module):
    def __init__(self, in_channels, out_channels, weight_scale, feature_scale):
        super(_MarginLinear, self).__init__()
        self.feature_scale = feature_scale
        self.weight_scale = weight_scale
        self.weight = Parameter(torch.zeros(out_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)


class MultiplicativeAngularMarginLinear(_MarginLinear):
    def __init__(self, in_channels, out_channels, m, weight_scale, feature_scale): 
        super(MultiplicativeAngularMarginLinear, self).__init__(
            in_channels, out_channels, weight_scale, feature_scale)
        assert isinstance(m, int)
        self.m = m
        self.register_buffer('m_choose_n_map', torch.zeros(self.m+1))
        self.register_buffer('k_map', torch.zeros(self.m+1))
        for i in range(self.m+1):
            n = k = i
            self.m_choose_n_map[i] = math.factorial(self.m) / math.factorial(n) / math.factorial(self.m-n)
            self.k_map[i] = math.cos(k * math.pi / self.m)
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]
    
    def forward(self, x):
        w = self.weight
        if self.weight_scale:
            w = w / torch.sqrt(w.pow(2).sum(1)).unsqueeze(1)
            if self.weight_scale != 1:
                w = w * self.weight_scale
        if self.feature_scale:
            x = x / torch.sqrt(w.pow(2).sum(1)).unsqueeze(1)
            if self.feature_scale != 1:
                x = x * self.weight_scale
        x_dot_wT = x.mm(w.transpose(0, 1))
        if self.m > 1:
            w_norm = torch.sqrt(w.pow(2).sum(1))
            x_norm = torch.sqrt(x.pow(2).sum(1))
            cos_theta = x_dot_wT / (x_norm.view(-1, 1) * w_norm.view(1, -1))
            cos_theta = cos_theta.clamp(-1, 1)
            angle = cos_theta.acos()
            k = ((self.m * angle) / math.pi).floor()
            sin_square_theta = 1 - cos_theta.pow(2)
            n = torch.arange(0, self.m // 2 + 1, dtype=cos_theta.dtype, device=cos_theta.device)
            cos_m_theta = pow(-1, n)[:, None] * self.m_choose_n_map[2*n.long()][:, None] * (cos_theta[:, None, :] ** (self.m - 2*n)[:, None]) * (sin_square_theta[:, None, :] ** n[:, None])
            cos_m_theta = torch.sum(cos_m_theta, 1)
            phi_theta = (pow(-1, k) * cos_m_theta - 2*k)
            f_m = phi_theta * x_norm.view(-1, 1) * w_norm.view(1, -1)
        return (x_dot_wT, f_m)

class AdditiveCosineMarginLinear(_MarginLinear):
    def __init__(self, in_channels, out_channels, m=4, weight_scale=None, feature_scale=None): 
        super(AdditiveCosineMarginLinear, self).__init__(
            in_channels, out_channels, weight_scale, feature_scale)
        self.m = m
    
    def forward(self, x):
        w = self.weight

        w_norm = torch.sqrt(w.pow(2).sum(1)).unsqueeze(1)
        w = w / w_norm 

        x_norm = torch.sqrt(x.pow(2).sum(1)).unsqueeze(1)
        x = x / x_norm 

        cos_theta = x.mm(w.transpose(0, 1))
        cos_theta = cos_theta.clamp(-1, 1)
        phi_cos_theta = cos_theta - self.m

        x_dot_wT = cos_theta
        f_m = phi_cos_theta
        if self.weight_scale is None:
            x_dot_wT = x_dot_wT * w_norm.view(1, -1)
            f_m = f_m * w_norm.view(1, -1) 
        else:
            if self.weight_scale != 1:
                x_dot_wT = x_dot_wT * self.weight_scale
                f_m = f_m * self.weight_scale

        if self.feature_scale is None:
            x_dot_wT = x_dot_wT * x_norm.view(1, -1)
            f_m = f_m * x_norm.view(1, -1) 
        else:
            if self.feature_scale != 1:
                x_dot_wT = x_dot_wT * self.feature_scale
                f_m = f_m * self.feature_scale

        return (x_dot_wT, f_m)

class AdditiveAngularMarginLinear(_MarginLinear):
    def __init__(self, in_channels, out_channels, base, gamma, power, lambda_min, 
                 m, **kwargs):
        super(AdditiveAngularMarginLinear, self).__init__(
            in_channels, out_channels, base, gamma, power, lambda_min, **kwargs)
        self.m = m
    
    def forward(self, x, y=None):
        pass

class CombinedMarginLinear(_MarginLinear):
    def __init__(self, in_channels, out_channels, base, gamma, power, lambda_min, 
                 m1, m2, m3, **kwargs):
        super(CombinedMarginLinear, self).__init__(
            in_channels, out_channels, base, gamma, power, lambda_min, **kwargs)
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
    
    def forward(self, x, y=None):
        pass
