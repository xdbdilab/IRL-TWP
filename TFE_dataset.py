from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.optim as optim


class TaskDataset(Dataset):

    def __init__(self, pickle_path, file_num_lower, file_num_upper, transform=None):
        super(TaskDataset).__init__()
        self.pickle_path = pickle_path
        self.file_num_lower = file_num_lower
        self.file_num_upper = file_num_upper
        self.transform = transform

    def __getitem__(self, index):
        data_path = self.pickle_path + str(index+self.file_num_lower) + '_pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data[0], data[1]

    def __len__(self):
        return self.file_num_upper - self.file_num_lower
