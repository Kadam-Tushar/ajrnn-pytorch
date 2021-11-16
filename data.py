import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ITSCDataSet(Dataset):

    def __init__(self, path, missing_frac=0.2):
        self.missing_frac = missing_frac
        arr = np.load_txt(path, delimiter=',')
        self.labels = arr[:, 0]
        self.data = arr[:, 1:, np.newaxis]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_tensor = torch.from_numpy(self.data[idx, :, :]),
        target_tensor = torch.from_numpy(self.labels[idx])

        mask_tensor = (torch.rand(data_tensor.shape) < (1-self.missing_frac))
        mask_tensor = mask_tensor.int()

        return (data_tensor, mask_tensor, target_tensor)
