import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime

from models import RegressionLearner as Learner
from utils import random_uniform, sine_function

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

    save_dir = "./trained_models/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    savepath = os.path.join(save_dir, 'regression_pretrain.pth')
    print("savepath =", savepath)

    lr_b = 5e-3
    print("lr_beta = {:.2e}".format(lr_b))

    optimizer = torch.optim.Adam(reg_learner.parameters(), lr=lr_b, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
    optimizer.zero_grad()

    # hyperparameters
    num_epochs = 100000
    num_tasks  = 200
    num_points = 5

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
        if epoch % 1000 == 0:
            with open('./trained_models/reg_pretrain_results.txt', 'a') as f:
                f.write("[{}] epoch {}: meta_learning_loss = {:.3e}".format(str(datetime.now()), epoch, mse_loss))
                
        optimizer.step()
        optimizer.zero_grad()

    print("finished pre-training")
    torch.save(reg_learner.state_dict(), savepath)

if __name__ == "__main__":
    regression_maml_exp()
