import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime
import pickle
from sklearn.utils import shuffle

from models.miniimagenet import LearnerConv

# --------------------- Experiment setting --------------------- #
use_gpu = True # set to False if there is no GPU available
if use_gpu:
    device = 'cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    device = 'cpu'

imagenet_learner = LearnerConv(N_way=5, device=device)

loadpath = "trained_models/imagenet_approx_5march_n5_k5_final.pt"
print("loadpath =", loadpath)
imagenet_learner.load_state_dict(torch.load(loadpath))
imagenet_learner.eval()

# --------------------- Load the MiniImageNet data --------------------- #
imagenet_test_path  = "data/miniimagenet.test.nparray.pk"
with open(imagenet_test_path, 'rb') as f: X_test = pickle.load(f, encoding="bytes")
X_test = np.transpose(X_test, (0,1,4,3,2))


X_test = np.transpose(X_test, (1,0,2,3,4))
X_test = shuffle(X_test, random_state=4)
X_test = np.transpose(X_test, (1,0,2,3,4))

# --------------------- MAML MiniImageNet experiment --------------------- #
def miniimagenet_maml_exp_eval():
    # hyperparameters
    num_tasks       = 5 # N
    num_points      = 5  # K
    num_grad_update = 10  # for evaluation
    batch_size      = num_tasks*num_points
    lr_a            = 0.01

    eval_num_points = 600-num_points # each char has 600 instaces
    eval_class_examples = 50
    eval_batch_size  = num_tasks * eval_class_examples
    eval_iter        = int(eval_num_points/eval_class_examples)

    num_eval_char   = X_test.shape[0]
    num_iterations  = int(num_eval_char/num_tasks)

    criterion = nn.NLLLoss(reduction='mean')
    optimizer = torch.optim.SGD(imagenet_learner.parameters(), lr=lr_a, momentum=0.0)
    optimizer.zero_grad()

    idx = 0
    count_correct_pred = 0
    count_total_pred   = 0


    for iter in range(num_iterations):
        # 1. for task_i consisting of characters of [idx, idx+num_tasks)
        imagenet_learner.load_state_dict(torch.load(loadpath))

        # 2. update the gradient 'num_grad_update' times
        X_batch = np.zeros((batch_size, 3, 84, 84))
        Y_batch = np.zeros((batch_size))

        for k in range(num_tasks):
            X_batch[k*num_points:(k+1)*num_points,:,:,:] = X_test[idx+k,:num_points,:,:,:]
            Y_batch[k*num_points:(k+1)*num_points] = k

        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
        Y_batch = torch.tensor(Y_batch, dtype=torch.long).to(device)

        for j in range(num_grad_update):
            # 2.2 compute gradient
            Y_pred = imagenet_learner(X_batch)
            # print(Y_pred.argmax(dim=-1))
            loss = criterion(Y_pred, Y_batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


        # 3. evaluation
        this_corr_pred = 0
        this_total_pred = 0
        ei = 0
        for _k in range(eval_iter):
            X_batch_eval = np.zeros((eval_batch_size, 3, 84, 84))
            Y_batch_eval = np.zeros((eval_batch_size))
            for k in range(num_tasks):
                X_batch_eval[k*eval_class_examples:(k+1)*eval_class_examples,:,:,:] = X_test[idx+k,num_points+ei:num_points+ei+eval_class_examples,:,:,:]
                Y_batch_eval[k*eval_class_examples:(k+1)*eval_class_examples] = k

            # try shuffling the rows for sanity check already!! --- no need for speed
            # X_batch_eval, Y_batch_eval = shuffle(X_batch_eval, Y_batch_eval, random_state=0)

            X_batch_eval = torch.tensor(X_batch_eval, dtype=torch.float32).to(device)
            Y_batch_eval = torch.tensor(Y_batch_eval, dtype=torch.long).to(device)

            Y_pred_eval = imagenet_learner(X_batch_eval)
            Y_pred_eval = Y_pred_eval.argmax(dim=-1)

            corr_pred  = (Y_batch_eval == Y_pred_eval).int().sum().item()
            total_pred = len(Y_batch_eval)

            this_corr_pred += corr_pred
            this_total_pred += total_pred

            ei += eval_class_examples # eval index

            print("eval {}/{}: Accuray = {:.3f}".format(_k, eval_iter, corr_pred/total_pred))

        count_correct_pred += this_corr_pred
        count_total_pred   += this_total_pred

        print("[{}] iteration {}/{}: Accuray = {:.3f}".format(str(datetime.now()), iter, num_iterations, this_corr_pred/this_total_pred))
        # import pdb; pdb.set_trace()

        idx += num_tasks

    print("PREDICTION ACCURACY = {}".format(count_correct_pred/count_total_pred))

miniimagenet_maml_exp_eval()
