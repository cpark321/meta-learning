import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime
from models import RegressionLearner as Learner
from utils import random_uniform, sine_function


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
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    else:
        device = 'cpu'

    reg_learner = Learner(hidden_size=40, device=device)
    mse_criterion = nn.MSELoss(reduction='mean')

    lr_b = 5e-3
    print("lr_beta = {:.2e}".format(lr_b))

    optimizer = torch.optim.Adam(reg_learner.parameters(), lr=lr_b, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
    optimizer.zero_grad()

    # hyperparameters
    num_epochs = 100000
    num_tasks  = 200
    num_points = 5 # K
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
        if epoch % 100 == 0:
            print("[{}] epoch {}: meta_learning_loss = {:.3e}".format(str(datetime.now()), epoch, meta_learning_loss))
            sys.stdout.flush()

        meta_learning_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 1000 == 0:
            with open('./trained_models/reg_maml_results.txt', 'a') as f:
                f.write("[{}] epoch {}: meta_learning_loss = {:.3e}".format(str(datetime.now()), epoch, meta_learning_loss))

    savepath = "trained_models/regression_maml.pth"
    print("saving a model at", savepath)
    torch.save(reg_learner.state_dict(), savepath)


if __name__ == "__main__":
    regression_maml_exp()
