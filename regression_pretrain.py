import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime

# ---------------- Learner class definition ----------------- #
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

# --------------------- Helper Functions --------------------- #
def random_uniform(a, b):
    return (b - a)*np.random.random_sample() + a
def sine_function(amp, phi, x):
    return amp*np.sin(x + phi)

# --------------------- MAML Regression experiment --------------------- #
def regression_maml_exp():
    # --------------------- Experiment setting --------------------- #
    use_gpu = True # set to False if there is no GPU available
    if use_gpu:
        device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    else:
        device = 'cpu'

    reg_learner = Learner(hidden_size=40, device=device)
    mse_criterion = nn.MSELoss(reduction='mean')

    savepath = "trained_models/reg_6march_pretrain.pt"
    print("savepath =", savepath)

    lr_b = 1e-2
    print("lr_beta = {:.2e}".format(lr_b))

    optimizer = torch.optim.Adam(reg_learner.parameters(), lr=lr_b, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
    optimizer.zero_grad()

    # hyperparameters
    num_epochs = 50000
    num_tasks  = 200
    num_points = 10

    for epoch in range(num_epochs):
        for task in range(num_tasks):
            # 1. sample amplitude & phase
            amp = random_uniform(0.1, 5.0)
            phi     = random_uniform(0, np.pi)

            X = [None for _ in range(num_points)]
            Y = [None for _ in range(num_points)]
            for j in range(num_points):
                X[j] = random_uniform(-5.0, 5.0)
                Y[j] = sine_function(amp, phi, X[j])

            # 2. compute gradient & do backprop
            Xtrain = torch.tensor(X).unsqueeze(-1).to(device)
            Ytrain = torch.tensor(Y).unsqueeze(-1).to(device)
            Ypred = reg_learner(Xtrain)
            mse_loss = mse_criterion(Ypred, Ytrain)
            mse_loss /= num_tasks
            mse_loss.backward()

        # 3. Update model's parameters
        if epoch % 10 == 0:
            print("[{}] epoch {}: mse_loss = {:.3e}".format(str(datetime.now()), epoch, mse_loss))
            sys.stdout.flush()

        optimizer.step()
        optimizer.zero_grad()

    print("finished pre-training")
    torch.save(reg_learner.state_dict(), savepath)

if __name__ == "__main__":
    regression_maml_exp()
