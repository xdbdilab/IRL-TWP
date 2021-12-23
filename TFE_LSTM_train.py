import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from TFE_dataset import TaskDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pickle
import numpy as np

TimeStep = 20
PredStep = 20
InputSize = 200
BatchSize = 32
LearningRate = 0.003
HiddenSize = 128
NumLayers = 4
NumClasses = 1


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(2)
# device = torch.device('cpu')

class task_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(task_LSTM, self).__init__()
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
        out = out[:, -PredStep:, :]
        # Decode the hidden state of the last time step
        out = torch.tanh(self.fc(out))
        return out, hidden_info


writer = SummaryWriter('runs/gru_20-20-0723')


train_set = TaskDataset('/27T/TE_TWP/task_usage_pickle/', 0, 1200000)
test_set = TaskDataset('/27T/TE_TWP/task_usage_pickle/', 1200000, 1300000)
train_loader = DataLoader(train_set, batch_size=BatchSize, shuffle=True, drop_last=True)
test_loader = DataLoader(train_set, batch_size=1)

net = task_LSTM(InputSize, HiddenSize, NumLayers, NumClasses).to(device)
mseloss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LearningRate)
iterations_per_epoch = int(len(train_set)/BatchSize)

for epoch in range(40):
    running_loss = 0.0
    for i, (x, y) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        input = x.type('torch.FloatTensor').to(device)
        target = y.type('torch.FloatTensor')
        target = target[:, -PredStep:, 0]
        target = target.view(BatchSize, PredStep, 1).to(device)
    
        output, _ = net.forward(input)
        loss = mseloss(output, target)

        loss.backward(retain_graph=True)

        running_loss += loss.item()
        optimizer.step()

        if i % 1000 == 999:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            writer.add_scalar('train_loss', running_loss / 1000, i + iterations_per_epoch * epoch)
            running_loss = 0.0
            pickle.dump(net, open('task_usage_model_0723-128-{}.p'.format(epoch), 'wb'))

test_loss = []
for i, (x, y) in enumerate(test_loader, 0):

    with torch.no_grad():
        input = x.type('torch.FloatTensor').to(device)
        target = y.type('torch.FloatTensor')
        target = target[:, -PredStep:, 0]
        target = target.view(1, PredStep, 1).to(device)
        # print(target)
        # print(input)
        output, _ = net.forward(input)
        loss = mseloss(output, target)

        running_loss += loss.item()

        if i % 1000 == 999:  # print every 1000 tests
            test_loss.append(running_loss/1000)
            print('[%5d] loss: %.5f' %
                  (i + 1, running_loss / 1000))
            writer.add_scalar('test_loss', running_loss / 1000, i)
            running_loss = 0.0

print(np.array(test_loss).mean())

writer.close()
