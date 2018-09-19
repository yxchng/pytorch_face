
import os
import cv2
import numpy as np
import torch.utils.data as data

class ImageList(data.Dataset):
    def __init__(self, data_dir, data_list, transform=None):
        self.data_list = data_list
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        f = open(data_list, 'r')
        for line in f:
            line = line.strip().split()
            self.img_paths.append(os.path.join(data_dir, line[0]))
            self.labels.append(int(line[1]))

    def num_classes(self):
        return len(np.unique(self.labels))
                
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        image  = cv2.imread(img_path)
        image = np.float32(image)
        if self.transform:
            image = self.transform(image)
        return image, label


