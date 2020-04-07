import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RegressionLearner(nn.Module):
    def __init__(self, hidden_size, device):
        super(RegressionLearner, self).__init__()
        self.device = device

        self.w1 = nn.Parameter(torch.randn(hidden_size, 1), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        self.w3 = nn.Parameter(torch.randn(1, hidden_size), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1, 1), requires_grad=True)

        self.to(device)

    def forward(self, x):
        # x is a one-dimension input [batch_size, 1]
        # note that the author also add normalisation layer...
        z1 = F.relu(torch.matmul(x,self.w1.T) + self.b1)
        z2 = F.relu(torch.matmul(z1,self.w2.T) + self.b2)
        z3 = torch.matmul(z2,self.w3.T) + self.b3
        return z3

    def forward_adapted_params(self, x, params):
        w1 = params['w1']
        b1 = params['b1']
        w2 = params['w2']
        b2 = params['b2']
        w3 = params['w3']
        b3 = params['b3']

        z1 = F.relu(torch.matmul(x,w1.T) + b1)
        z2 = F.relu(torch.matmul(z1,w2.T) + b2)
        z3 = torch.matmul(z2,w3.T) + b3
        return z3
