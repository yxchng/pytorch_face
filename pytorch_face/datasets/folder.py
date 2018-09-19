import os
import cv2
import numpy as np
import torch.utils.data as data

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, removed_ids=None):
        self.root = root
        self.transform = transform
        self.imagepaths = []
        self.labels = []

        self.removed_idpaths = []
        if removed_ids:
            f = open(removed_ids, 'r')
            for line in f:
                line = line.strip()
                idpath = os.path.join(root, line)
                self.removed_idpaths.append(idpath)
            
        idx = -1
        for _id in os.listdir(self.root):
            idpath = os.path.join(self.root, _id)
            if self.removed_idpaths:
                if idpath in self.removed_idpaths:
                    continue
            idx += 1
            for _, _, filenames in os.walk(idpath):
                for filename in filenames:
                    imagepath = os.path.join(idpath, filename)
                    if imagepath.endswith('.jpg') or imagepath.endswith('.png'):
                        self.imagepaths.append(imagepath)
                        self.labels.append(idx)

    def num_classes(self):
        return len(np.unique(self.labels))
                
    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, index):
        imagepath = self.imagepaths[index]
        label = self.labels[index]
        image  = cv2.imread(imagepath)
        #bgr to rgb
        #image = image[:, :, (2, 1, 0)]
        if self.transform:
            image = self.transform(image)
        return image, label


