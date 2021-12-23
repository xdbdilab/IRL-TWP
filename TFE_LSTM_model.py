import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from TFE_dataset import TaskDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pickle

TimeStep = 10
PredStep = 10
InputSize = 200
BatchSize = 32
LearningRate = 0.001
HiddenSize = 64
NumLayers = 2
NumClasses = 1


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class task_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(task_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        hidden_info = out[:, -1, :]

        # Decode the hidden state of the last time step
        out = torch.tanh(self.fc(out))
        return out, hidden_info

