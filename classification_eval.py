import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime
import pickle
# from sklearn.utils import shuffle

from models.omniglot import LearnerConv

# --------------------- Experiment setting --------------------- #
use_gpu = True # set to False if there is no GPU available
if use_gpu:
    device = 'cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    device = 'cpu'

omniglot_learner = LearnerConv(N_way=5, device=device)

loadpath = "trained_models/omniglot_model_exp0.pt"
print("loadpath =", loadpath)
omniglot_learner.load_state_dict(torch.load(loadpath))
omniglot_learner.eval()

# --------------------- Load the Omniglot data --------------------- #
omniglot_np_path = "data/omniglot.nparray.pk"
with open(omniglot_np_path, 'rb') as f:
    X_data = pickle.load(f, encoding="bytes")
np.random.seed(28)
np.random.shuffle(X_data)
X_train = X_data[:1200,:,:,:]
X_test  = X_data[1200:,:,:,:]

# --------------------- MAML Omniglot experiment --------------------- #
def omniglot_maml_exp_eval():
    # hyperparameters
    num_tasks       = 5 # N
    num_points      = 5 # K
    num_grad_update = 3 # for evaluation
    batch_size      = num_tasks*num_points
    lr_a            = 0.4

    num_eval_char   = X_test.shape[0]
    num_iterations  = int(num_eval_char/num_tasks)

    criterion = nn.NLLLoss(reduction='mean')
    optimizer = torch.optim.SGD(omniglot_learner.parameters(), lr=lr_a, momentum=0.0)
    optimizer.zero_grad()

    idx = 0
    count_correct_pred = 0
    count_total_pred   = 0
    for iter in range(num_iterations):
        # 1. for task_i consisting of characters of [idx, idx+num_tasks)
        omniglot_learner.load_state_dict(torch.load(loadpath))

        # 2. update the gradient 'num_grad_update' times
        e_idx = 0 # element_idx
        for j in range(num_grad_update):

            X_batch = np.zeros((batch_size, 28, 28))
            Y_batch = np.zeros((batch_size))

            for k in range(num_tasks):
                X_batch[k*num_points:(k+1)*num_points,:,:] = X_test[idx+k,e_idx:e_idx+num_points,:,:]
                Y_batch[k*num_points:(k+1)*num_points] = k


            # 2.2 compute gradient
            X_batch = torch.tensor(X_batch, dtype=torch.float32).unsqueeze(1).to(device)
            Y_batch = torch.tensor(Y_batch, dtype=torch.long).to(device)
            Y_pred = omniglot_learner(X_batch)

            loss = criterion(Y_pred, Y_batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            e_idx += num_points

        # 3. evaluation
        X_batch_eval = np.zeros((batch_size, 28, 28))
        Y_batch_eval = np.zeros((batch_size))
        for k in range(num_tasks):
            X_batch_eval[k*num_points:(k+1)*num_points,:,:] = X_test[idx+k,e_idx:e_idx+num_points,:,:]
            Y_batch_eval[k*num_points:(k+1)*num_points] = k

        # try shuffling the rows for sanity check already!! --- no need for speed
        # X_batch_eval, Y_batch_eval = shuffle(X_batch_eval, Y_batch_eval, random_state=0)

        X_batch_eval = torch.tensor(X_batch_eval, dtype=torch.float32).unsqueeze(1).to(device)
        Y_batch_eval = torch.tensor(Y_batch_eval, dtype=torch.long).to(device)

        Y_pred_eval = omniglot_learner(X_batch_eval)
        Y_pred_eval = Y_pred_eval.argmax(dim=-1)

        count_correct_pred += (Y_batch_eval == Y_pred_eval).int().sum().item()
        count_total_pred   += len(Y_batch_eval)

        print("[{}] iteration {}/{}: ".format(str(datetime.now()), iter, num_iterations))

        idx += num_tasks

    print("PREDICTION ACCURACY = {}".format(count_correct_pred/count_total_pred))

omniglot_maml_exp_eval()
