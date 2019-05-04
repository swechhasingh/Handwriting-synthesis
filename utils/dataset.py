import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class HandwritingDataset(Dataset):
    """Handwriting dataset."""

    def __init__(self, inputdata, targetdata, mask):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputdata = inputdata
        self.targetdata = targetdata
        self.mask = mask

    def __len__(self):
        return self.inputdata.shape[0]

    def __getitem__(self, idx):
        input_seq = torch.from_numpy(self.inputdata[idx])
        target = torch.from_numpy(self.targetdata[idx])
        mask = torch.from_numpy(self.mask[idx])
        sample = (input_seq, target, mask)
        return sample
