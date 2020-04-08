import torch
import torch.nn as nn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from omniglotDataset import OmniglotDataset

from models import LearnerConv

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--n_way', type=int, required=True)
parser.add_argument('-k', '--k_shot', type=int, required=False, default=None)
parser.add_argument('-c', '--no_cuda', required=False, default=None)
parser.add_argument('--no_iter', default= 50000, type= int, required=False, help='number of epochs')
parser.add_argument('--lr', default= 0.1, type=float , required=False, help='learning rate')

args = parser.parse_args()

num_tasks = args.n_way
num_points = args.k_shot
num_iterations = args.no_iter

save_path = os.path.join('./saves/', f'{num_tasks}-way-{num_points}-shot')

if not os.path.exists(save_path):
    os.makedirs(save_path)

device_type='cuda'

if args.no_cuda is not None:
    device_type = 'cuda:'+args.no_cuda

device = torch.device(device_type if torch.cuda.is_available() else 'cpu')



Odset = OmniglotDataset(root='./', download=True)
X_data = Odset.getOmniglotArray()

np.random.shuffle(X_data)
X_train = X_data[:1200,:,:,:]
X_test  = X_data[1200:,:,:,:]

batch_size      = num_tasks*num_points
metabatch_size  = 32 # 32 tasks --- the number of tasks sampled per meta-update
                     #          --- each task is an N-way, K-shot classification problem

lr_a = args.lr
num_grad_update = 1

print("N={}".format(num_tasks))
print("K={}".format(num_points))
print("metabatch_size={}".format(metabatch_size))
print("lr_a={}".format(lr_a))
print("num_grad_update={}".format(num_grad_update))

omniglot_learner = LearnerConv(N_way=num_tasks, device=device)
# print(omniglot_learner)

lr_b = 1e-4
print("lr_beta = {:.2e}".format(lr_b))

criterion = nn.NLLLoss(reduction='mean')

optimizer = torch.optim.Adam(omniglot_learner.parameters(), lr=lr_b, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
optimizer.zero_grad()

for iter in range(num_iterations):
    # 1. sample batch of tasks Ti ~ p(T)
    tasks = [None for _ in range(metabatch_size)]
    for _i in range(metabatch_size):
        tasks[_i] = np.random.randint(low=0, high=1200, size=num_tasks)

    # 2. for each task Ti
    meta_learning_loss = 0
    for task in tasks:

        # copy current model weights to fast_weights
        fast_weights = omniglot_learner.copy_model_weights()

        X_batch_a = np.zeros((batch_size, 28, 28))
        Y_batch_a = np.zeros((batch_size))
        X_batch_b = np.zeros((batch_size, 28, 28))
        Y_batch_b = np.zeros((batch_size))

        # 2.1 sample K datapoints from Ti
        for j1, char_id in enumerate(task):
            instances = np.random.randint(low=0, high=20, size=num_points)
            for j2, ins in enumerate(instances):
                X_batch_a[j1*num_points+j2,:,:] = X_train[char_id,ins,:,:]
                Y_batch_a[j1*num_points+j2] = j1
        X_batch_a = torch.tensor(X_batch_a, dtype=torch.float32).unsqueeze(1).to(device)
        Y_batch_a = torch.tensor(Y_batch_a, dtype=torch.long).to(device)
        # 2.2 compute gradient (multiple steps)
        for grad_update_iter in range(num_grad_update):
            Y_pred = omniglot_learner.forward_fast_weights(X_batch_a, fast_weights)
            train_loss = criterion(Y_pred, Y_batch_a)
            # print(train_loss)

            grad = torch.autograd.grad(train_loss, fast_weights, create_graph=True)
            fast_weights = omniglot_learner.update_fast_grad(fast_weights, grad, lr_a)

        # 2.3 sample K datapoints from Ti --- for meta-update step
        for j1, char_id in enumerate(task):
            instances = np.random.randint(low=0, high=20, size=num_points)
            for j2, ins in enumerate(instances):
                X_batch_b[j1*num_points+j2,:,:] = X_train[char_id,ins,:,:]
                Y_batch_b[j1*num_points+j2] = j1

        # 3. meta-update step
        X_batch_b = torch.tensor(X_batch_b, dtype=torch.float32).unsqueeze(1).to(device)
        Y_batch_b = torch.tensor(Y_batch_b, dtype=torch.long).to(device)

        Y_pred = omniglot_learner.forward_fast_weights(X_batch_b, fast_weights)
        meta_loss = criterion(Y_pred, Y_batch_b)
        meta_learning_loss += meta_loss

    # 4. Backpropagation to update model's parameters
    meta_learning_loss /= num_tasks
    
#     if iter % 100 == 0:
#         print("[{}] iteration {}: meta_learning_loss = {:.5f}".format(str(datetime.now()), iter, meta_learning_loss))
#         savepath = "trained_models/omniglot_4marchv2_n{}_k{}_iter{}.pt".format(num_tasks, num_points, iter)
#         print("saving a model at", savepath)
#         torch.save(omniglot_learner.state_dict(), savepath)
#         sys.stdout.flush()

    meta_learning_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if iter % 1000 == 0:
        with open(os.path.join(save_path, f'training-log-lr{lr_a}.txt'), 'a') as f:
            f.write("[{}] iter {}: meta_learning_loss = {:.3e}\n".format(str(datetime.now()), iter, meta_learning_loss))

savepath = "trained_models/omniglot_n{}_k{}_lr{}_final.pth".format(num_tasks, num_points, lr_a)
print("saving a model at", savepath)
torch.save(omniglot_learner.state_dict(), savepath)

print("finished maml training")


