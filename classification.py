import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime
import pickle

from models.omniglot import LearnerConv

# --------------------- Experiment setting --------------------- #
use_gpu = True # set to False if there is no GPU available
if use_gpu:
    if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
        print('running on the stack...')
        cuda_device = os.environ['X_SGE_CUDA_DEVICE']
        print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    else:
        # pdb.set_trace()
        print('running locally...')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' # choose the device (GPU) here
    device = 'cuda'
else:
    device = 'cpu'
print("device = {}".format(device))

omniglot_learner = LearnerConv(N_way=20, device=device)
print(omniglot_learner)


lr_b = 1e-2
print("lr_beta = {:.2e}".format(lr_b))

criterion = nn.NLLLoss(reduction='mean')

optimizer = torch.optim.Adam(omniglot_learner.parameters(), lr=lr_b, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
optimizer.zero_grad()

# --------------------- Load the Omniglot data --------------------- #
omniglot_np_path = "data/omniglot.nparray.pk"
with open(omniglot_np_path, 'rb') as f:
    X_data = pickle.load(f, encoding="bytes")
np.random.seed(28)
np.random.shuffle(X_data)
X_train = X_data[:1200,:,:,:]
X_test  = X_data[1200:,:,:,:]

# --------------------- MAML Omniglot experiment --------------------- #
def omniglot_maml_exp():
    # hyperparameters
    num_iterations  = 60000
    num_tasks       = 20 # N
    num_points      = 5 # K
    batch_size      = num_tasks*num_points
    metabatch_size  = 16 # 32 tasks --- the number of tasks sampled per meta-update
                         #          --- each task is an N-way, K-shot classification problem
    lr_a = 0.1

    num_grad_update = 5

    print("N={}".format(num_tasks))
    print("K={}".format(num_points))
    print("metabatch_size={}".format(metabatch_size))
    print("lr_a={}".format(lr_a))
    print("num_grad_update={}".format(num_grad_update))

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

            for grad_update_iter in range(num_grad_update):
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

                # 2.2 compute gradient
                X_batch_a = torch.tensor(X_batch_a, dtype=torch.float32).unsqueeze(1).to(device)
                Y_batch_a = torch.tensor(Y_batch_a, dtype=torch.long).to(device)

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
        if iter % 10 == 0:
            print("[{}] iteration {}: meta_learning_loss = {:.5f}".format(str(datetime.now()), iter, meta_learning_loss))
            sys.stdout.flush()

        meta_learning_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter % 5000 == 0:
            savepath = "trained_models/omniglot_24feb_n{}_k{}_iter{}.pt".format(num_tasks, num_points, iter)
            print("saving a model at", savepath)
            torch.save(omniglot_learner.state_dict(), savepath)

    print("finished maml training")

omniglot_maml_exp()
