import random
import torch
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img -= self.mean
        img /= self.std
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.fliplr(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class ToTensor(object):
    def __call__(self, img):
        tensor = torch.from_numpy(img.copy()).permute(2, 0, 1)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToDevice(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, tensor):
        tensor = tensor.to(self.device)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(device={})'.format(self.device)
