import os
import numpy as np
import torch
from torch.utils.data import Dataset

def transfer_labels(labels):
	#some labels are [1,2,4,11,13] and is transfer to standard label format [0,1,2,3,4]
	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = labels.shape[0]

	for i in range(num_samples):
		new_label = np.argwhere( labels[i] == indexes )[0][0]
		labels[i] = new_label
	return labels, num_classes

class ITSCDataset(Dataset):

    def __init__(self, path, missing_frac=0.2):
        self.missing_frac = missing_frac
        arr = np.loadtxt(path, dtype=np.float32, delimiter=',')
        self.labels = arr[:, 0].astype(np.long)
        self.labels, _ = transfer_labels(self.labels)
        self.data = arr[:, 1:, np.newaxis]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_tensor = torch.from_numpy(self.data[idx, :, :])
        target = self.labels[idx]

        mask_tensor = (torch.rand(data_tensor.shape) < (1-self.missing_frac))
        mask_tensor = mask_tensor.int()

        return (data_tensor, mask_tensor, target)
