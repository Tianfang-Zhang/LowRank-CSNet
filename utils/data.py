import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import scipy.io as scio

__all__ = ['CsTrain']


class CsTrain(Data.Dataset):
    def __init__(self, mat_path):
        print('...loading data from: {}'.format(mat_path))
        mat_data = scio.loadmat(mat_path)
        self.data = mat_data['labels']
        self.win_size = int(np.sqrt(self.data.shape[1]))

    def __getitem__(self, index):
        img = torch.Tensor(self.data[index, :]).float()
        img = img.view(1, self.win_size, self.win_size)
        return img

    def __len__(self):
        return len(self.data)
