import torch
import torch.nn as nn
import torch.optim as optim
from TFE_dataset import TaskDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pickle
import numpy as np

step_len = 100
PredStep = 1
InputSize = 202
BatchSize = 32
LearningRate = 0.003
HiddenSize = 128
NumLayers = 4
NumClasses = 1


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class task_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(task_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, InputSize)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        hidden_info = out[:, -1, :]
        out = out[:, -PredStep:, :]
        # Decode the hidden state of the last time step
        out = torch.tanh(self.fc(out))
        return out, hidden_info


writer = SummaryWriter('runs/lstm_20-20-0813')


train_set = TaskDataset('/27T/TE_TWP/state-action_pickle_lstm/', 0, 90000)
train_loader = DataLoader(train_set, batch_size=BatchSize, shuffle=True, drop_last=True)

net = task_LSTM(InputSize, HiddenSize, NumLayers, NumClasses).to(device)
mseloss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LearningRate)
iterations_per_epoch = int(len(train_set)/BatchSize)
print(iterations_per_epoch)

for epoch in range(100):
    running_loss = 0.0
    for i, (x, y) in enumerate(train_loader, 0):
        # print(x, x.shape)
        # print(y, y.shape)
        x_in = x[:, range(InputSize)]
        ac_loss = 0.0
        for step in range(step_len):
            optimizer.zero_grad()
            input = x_in.type('torch.FloatTensor').to(device)
            real = []
            if step != step_len-1:
                real = x[:, (step+1) * InputSize]
            else:
                real = y[:, 0]
            target = real.type('torch.FloatTensor')
            target = target.view(BatchSize, 1).to(device)
            output, _ = net.forward(input)
            loss = mseloss(output, target)
            loss.backward(retain_graph=True)
            ac_loss += loss.item()
            if step != step_len-1:
                x_in = x[:, (step+1)*InputSize:(step+2)*InputSize]
                x_in = np.array(x_in)
                x_in = torch.from_numpy(x_in)
            optimizer.step()
        avg_loss = ac_loss/step_len
        running_loss += avg_loss
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 100))
            writer.add_scalar('train_loss', running_loss / 100, i + iterations_per_epoch * epoch)
            running_loss = 0.0
            torch.save(net.state_dict(), 'lstm_alibaba_0813')

writer.close()
