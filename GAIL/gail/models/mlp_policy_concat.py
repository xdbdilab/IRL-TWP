import torch
import torch.nn as nn
from utils.math import *


class Policy(nn.Module):
    def __init__(self, usage_state_dim, task_state_dim, action_dim, usage_hidden_size=(128, 128), task_hidden_size=(128,128),
                 output_hidden_size=128, activation='relu', log_std=0, device='cpu'):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.device = device
        self.usage_state_dim = usage_state_dim
        self.usage_affine_layers = nn.ModuleList()
        usage_last_dim = usage_state_dim
        for nh in usage_hidden_size:
            self.usage_affine_layers.append(nn.Linear(usage_last_dim, nh))
            usage_last_dim = nh
        self.task_state_dim = task_state_dim
        self.task_affine_layers = nn.ModuleList()
        task_last_dim = task_state_dim
        for nh in task_hidden_size:
            self.task_affine_layers.append(nn.Linear(task_last_dim, nh))
            task_last_dim = nh
        self.action_mean = nn.Linear(usage_last_dim+task_last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):

        usage_x = torch.take(x.clone(), torch.arange(self.usage_state_dim).to(x.device))
        usage_x = usage_x.view((-1, self.usage_state_dim))
        for affine in self.usage_affine_layers:
            usage_x = self.activation(affine(usage_x))
        task_x = torch.take(x.clone(), torch.arange(self.usage_state_dim, self.usage_state_dim+self.task_state_dim).to(x.device))
        task_x = task_x.view((-1, self.task_state_dim))
        for affine in self.task_affine_layers:
            task_x = self.activation(affine(task_x))
        x = torch.cat((usage_x, task_x), dim=1)
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


