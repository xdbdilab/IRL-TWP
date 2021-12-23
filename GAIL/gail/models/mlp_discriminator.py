import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, 1)


    def forward(self, x):
        for affine in self.affine_layers:
            x = affine(x)
            if affine != self.affine_layers[-1]:
                x = self.activation(x)
        return x
