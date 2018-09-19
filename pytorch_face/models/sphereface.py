import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from pytorch_face.layers import *

__all__ = ['SphereFace', 'sphereface20', 'sphereface64']

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, anchor=None):
        super(BasicBlock, self).__init__()
        self.anchor = anchor
        self.conv1 = conv3x3(inplanes, planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes)
        self.prelu2 = nn.PReLU(planes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)

    def forward(self, x):
        if self.anchor is not None:
            x = self.anchor(x)
        residual = x
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x += residual
        return x

class SphereFace(nn.Module):
    def __init__(self, block, layers, num_classes=None, margin_type=None, margin_parameters=None, is_deploy=False):
        self.inplanes = 3
        self.is_deploy = is_deploy
        self.num_classes = num_classes
        self.margin_type = margin_type
        self.margin_parameters = margin_parameters

        super(SphereFace, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.fc1 = nn.Linear(512*7*6, 512)
        if margin_type == "multiplicative_angular":
            self.margin_linear = MultiplicativeAngularMarginLinear(512, num_classes, 
                                                                   margin_parameters['margin'],  
                                                                   margin_parameters['weight_scale'], 
                                                                   margin_parameters['feature_scale'])
        elif margin_type == "additive_cosine":
            self.margin_linear = AdditiveCosineMarginLinear(512, num_classes, 
                                                            margin_parameters['margin'] if margin_parameters['margin'] else 0.35,
                                                            margin_parameters['weight_scale'] if margin_parameters['weight_scale'] else 1, 
                                                            margin_parameters['feature_scale'] if margin_parameters['feature_scale'] else 64)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def _make_layer(self, block, planes, blocks):
        anchor = []
        conv = conv3x3(self.inplanes, planes, stride=2, bias=True)
        nn.init.xavier_normal_(conv.weight)
        nn.init.constant_(conv.bias, 0)
        anchor.append(conv)
        anchor.append(nn.PReLU(planes))
        anchor = nn.Sequential(*anchor) 

        layers = []
        self.inplanes = planes 
        layers.append(block(self.inplanes, planes, anchor))

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        feature = self.fc1(x)

        if self.is_deploy:
            return feature
        else:
            (x_dot_wT, f_m) = self.margin_linear(feature)
            return (x_dot_wT, f_m)

def sphereface20(**kwargs):
    model = SphereFace(BasicBlock, [1, 2, 4, 1], **kwargs)
    return model

def sphereface64(**kwargs):
    model = SphereFace(BasicBlock, [3, 8, 16, 3], **kwargs)
    return model
