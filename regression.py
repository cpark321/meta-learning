import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class Learner(nn.Module):
    def __init__(self, hidden_size, device):
        super(Learner, self).__init__()
        self.device = device

        self.w1 = nn.Parameter(torch.randn(hidden_size, 1), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        self.w3 = nn.Parameter(torch.randn(1, hidden_size), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1, 1), requires_grad=True)

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
        b4 = params['b3']

        z1 = F.relu(torch.matmul(x,w1.T) + b1)
        z2 = F.relu(torch.matmul(z1,w2.T) + b2)
        z3 = torch.matmul(z2,w3.T) + b3
        return z3

def random_uniform(a, b):
    return (b - a)*np.random.random_sample() + a
def sine_function(amp, phi, x):
    return amp*np.sin(x + phi)

reg_learner = Learner(hidden_size=40, device='cpu')
mse_criterion = nn.MSELoss(reduction='mean')

def regression_maml_exp():
    # hyperparameters
    num_tasks  = 20
    num_points = 5 # K
    lr_a = 0.01
    lr_b = 0.01

    # 1. sample batch of tasks Ti ~ p(T)
    tasks     = []
    for i in range(num_tasks):
        amplitude = random_uniform(0.1, 5.0)
        phase     = random_uniform(0, np.pi)
        tasks.append((amplitude, phase))

    # 2. for each task Ti
    valid_data = []
    gradients  = []
    for task in tasks:
        # 2.1 sample K datapoints from Ti
        amp = task[0]
        phi = task[1]
        X = [None for _ in range(num_points)]
        Y = [None for _ in range(num_points)]
        for j in range(num_points):
            X[j] = random_uniform(-5.0, 5.0)
            Y[j] = sine_function(amp, phi, X[j])
        # 2.2 compute gradient
        Xtrain = torch.tensor(X).unsqueeze(-1)
        Ytrain = torch.tensor(Y).unsqueeze(-1)
        Ypred = reg_learner(Xtrain)
        mse_loss = mse_criterion(Ypred, Ytrain)

        grad = torch.autograd.grad(mse_loss, reg_learner.parameters())
        gradients.append(grad)

        # 2.3 sample validation datapoints for this task
        Xi_val = [None for _ in range(num_points)]
        Yi_val = [None for _ in range(num_points)]
        for j in range(num_points):
            Xi_val[j] = random_uniform(-5.0, 5.0)
            Yi_val[j] = sine_function(amp, phi, Xi_val[j])
        valid_data.append((Xi_val, Yi_val))

    # 3. meta-update step
    for i in range(len(task)):
        Xi_val, Yi_val = valid_data[i]
        grad = gradients[i]
        params = {
            'w1': reg_learner.w1 - lr_a*grad[0], 'b1': reg_learner.b1 - lr_a*grad[1],
            'w2': reg_learner.w2 - lr_a*grad[2], 'b2': reg_learner.b2 - lr_a*grad[3],
            'w3': reg_learner.w3 - lr_a*grad[4], 'b3': reg_learner.b3 - lr_a*grad[5],
        }
        pdb.set_trace()
        reg_learner.forward_adapted_params()

regression_maml_exp()
