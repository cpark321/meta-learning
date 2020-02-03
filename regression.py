import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import os
import sys
from datetime import datetime

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

def random_uniform(a, b):
    return (b - a)*np.random.random_sample() + a
def sine_function(amp, phi, x):
    return amp*np.sin(x + phi)

use_gpu = False
if use_gpu:
    device = 'cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    device = 'cpu'

reg_learner = Learner(hidden_size=40, device=device)
mse_criterion = nn.MSELoss(reduction='mean')

savepath = "trained_models/reg_model_exp1.pt"
print("savepath =", savepath)

lr_b = 1e-2
print("lr_beta = {:.2e}".format(lr_b))

optimizer = torch.optim.Adam(reg_learner.parameters(), lr=lr_b, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
optimizer.zero_grad()

def regression_maml_exp():
    # hyperparameters
    num_epochs = 50000
    num_tasks  = 200
    num_points = 10 # K
    lr_a = 0.01

    for epoch in range(num_epochs):
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
            Xtrain = torch.tensor(X).unsqueeze(-1).to(device)
            Ytrain = torch.tensor(Y).unsqueeze(-1).to(device)
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
        meta_learning_loss = 0
        for i in range(len(tasks)):
            Xi_val, Yi_val = valid_data[i]
            grad = gradients[i]
            fast_params = {
                'w1': reg_learner.w1 - lr_a*grad[0], 'b1': reg_learner.b1 - lr_a*grad[1],
                'w2': reg_learner.w2 - lr_a*grad[2], 'b2': reg_learner.b2 - lr_a*grad[3],
                'w3': reg_learner.w3 - lr_a*grad[4], 'b3': reg_learner.b3 - lr_a*grad[5],
            }
            Xmeta = torch.tensor(Xi_val).unsqueeze(-1).to(device)
            Ymeta = torch.tensor(Yi_val).unsqueeze(-1).to(device)
            Ypred = reg_learner.forward_adapted_params(Xmeta, fast_params)
            meta_learning_loss += mse_criterion(Ypred, Ymeta)
            # print(i, meta_learning_loss)

        # 4. Backpropagation to update model's parameters
        meta_learning_loss /= num_tasks
        if epoch % 10 == 0:
            print("[{}] epoch {}: meta_learning_loss = {:.3e}".format(str(datetime.now()), epoch, meta_learning_loss))
            sys.stdout.flush()

        meta_learning_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("finished maml training")
    torch.save(reg_learner.state_dict(), savepath)

regression_maml_exp()
