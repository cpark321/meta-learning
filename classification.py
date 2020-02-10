import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import os
import sys
from datetime import datetime
import pickle

class Learner(nn.Module):
    def __init__(self, N_way, device):
        super(Learner, self).__init__()
        self.device = device
        self.N_way  = N_way
        self.conv1  = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.bnorm1 = nn.BatchNorm2d(num_features=64, track_running_stats=False)

        self.conv2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.bnorm2 = nn.BatchNorm2d(num_features=64, track_running_stats=False)

        self.conv3  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.bnorm3 = nn.BatchNorm2d(num_features=64, track_running_stats=False)

        self.conv4  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.bnorm4 = nn.BatchNorm2d(num_features=64, track_running_stats=False)

        self.fc = nn.Linear(in_features=64, out_features=N_way, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.to(device)

    def forward(self, x):
        # x = [batch_size, 1, 28, 28]

        z1 = F.relu(self.bnorm1(self.conv1(x)))
        z2 = F.relu(self.bnorm2(self.conv2(z1)))
        z3 = F.relu(self.bnorm3(self.conv3(z2)))
        z4 = F.relu(self.bnorm4(self.conv4(z3)))

        z4 = z4.view(-1, 64)
        out = self.logsoftmax(self.fc(z4))

        return out

    def forward_fast_weights(self, x, grad, lr_a):
        # x = [batch_size, 1, 28, 28]

        conv1_w  = self.conv1.weight  - lr_a * grad[0]
        conv1_b  = self.conv1.bias    - lr_a * grad[1]
        bnorm1_w = self.bnorm1.weight - lr_a * grad[2]
        bnorm1_b = self.bnorm1.bias   - lr_a * grad[3]

        conv2_w  = self.conv2.weight  - lr_a * grad[4]
        conv2_b  = self.conv2.bias    - lr_a * grad[5]
        bnorm2_w = self.bnorm2.weight - lr_a * grad[6]
        bnorm2_b = self.bnorm2.bias   - lr_a * grad[7]

        conv3_w  = self.conv3.weight  - lr_a * grad[8]
        conv3_b  = self.conv3.bias    - lr_a * grad[9]
        bnorm3_w = self.bnorm3.weight - lr_a * grad[10]
        bnorm3_b = self.bnorm3.bias   - lr_a * grad[11]

        conv4_w  = self.conv4.weight  - lr_a * grad[12]
        conv4_b  = self.conv4.bias    - lr_a * grad[13]
        bnorm4_w = self.bnorm4.weight - lr_a * grad[14]
        bnorm4_b = self.bnorm4.bias   - lr_a * grad[15]

        fc_w     = self.fc.weight     - lr_a * grad[16]
        fc_b     = self.fc.bias       - lr_a * grad[17]

        z1 = F.conv2d(x, conv1_w, conv1_b, stride=2, padding=0)
        z1 = F.batch_norm(z1, running_mean=self.bnorm1.running_mean, running_var=self.bnorm1.running_var, weight=bnorm1_w, bias=bnorm1_b, training=True) # how about training=True??
        z1 = F.relu(z1)

        z2 = F.conv2d(z1, conv2_w, conv2_b, stride=2, padding=0)
        z2 = F.batch_norm(z2, running_mean=self.bnorm2.running_mean, running_var=self.bnorm2.running_var, weight=bnorm2_w, bias=bnorm2_b, training=True) # how about training=True??
        z2 = F.relu(z2)

        z3 = F.conv2d(z2, conv3_w, conv3_b, stride=2, padding=0)
        z3 = F.batch_norm(z3, running_mean=self.bnorm3.running_mean, running_var=self.bnorm3.running_var, weight=bnorm3_w, bias=bnorm3_b, training=True) # how about training=True??
        z3 = F.relu(z3)

        z4 = F.conv2d(z3, conv4_w, conv4_b, stride=1, padding=0)
        z4 = F.batch_norm(z4, running_mean=self.bnorm4.running_mean, running_var=self.bnorm4.running_var, weight=bnorm4_w, bias=bnorm4_b, training=True) # how about training=True??
        z4 = F.relu(z4)

        z4 = z4.view(-1, 64)
        z4 = torch.matmul(z4, fc_w.T) + fc_b
        out = F.log_softmax(z4, dim=-1)
        return out


# --------------------- Experiment setting --------------------- #
use_gpu = True # set to False if there is no GPU available
if use_gpu:
    device = 'cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    device = 'cpu'

omniglot_learner = Learner(N_way=5, device=device)
criterion = nn.NLLLoss(reduction='mean')

savepath = "trained_models/omniglot_model_exp0.pt"
print("savepath =", savepath)

lr_b = 1e-2
print("lr_beta = {:.2e}".format(lr_b))

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
    num_tasks       = 5 # N
    num_points      = 5 # K
    batch_size      = num_tasks*num_points
    metabatch_size  = 32 # 32 tasks --- the number of tasks sampled per meta-update
                         #          --- each task is an N-way, K-shot classification problem
    lr_a = 0.4

    for iter in range(num_iterations):
        # 1. sample batch of tasks Ti ~ p(T)
        tasks = [None for _ in range(metabatch_size)]
        for _i in range(metabatch_size):
            tasks[_i] = np.random.randint(low=0, high=1200, size=num_tasks)

        # 2. for each task Ti
        # valid_data = []
        # gradients  = []
        meta_learning_loss = 0
        for task in tasks:
            X_batch_a = np.zeros((batch_size, 28, 28))
            Y_batch_a = np.zeros((batch_size))
            X_batch_b = np.zeros((batch_size, 28, 28))
            Y_batch_b = np.zeros((batch_size))

            # 2.1 sample K datapoints from Ti
            # 2.3 sample K datapoints from Ti --- for meta-update step
            for j1, char_id in enumerate(task):
                instances = np.random.randint(low=0, high=20, size=num_points)
                for j2, ins in enumerate(instances):
                    X_batch_a[j1*num_points+j2,:,:] = X_train[char_id,ins,:,:]
                    Y_batch_a[j1*num_points+j2] = j1
                instances = np.random.randint(low=0, high=20, size=num_points)
                for j2, ins in enumerate(instances):
                    X_batch_b[j1*num_points+j2,:,:] = X_train[char_id,ins,:,:]
                    Y_batch_b[j1*num_points+j2] = j1

            # valid_data.append((X_batch_b, Y_batch_b))

            # 2.2 compute gradient
            X_batch_a = torch.tensor(X_batch_a, dtype=torch.float32).unsqueeze(1).to(device)
            Y_batch_a = torch.tensor(Y_batch_a, dtype=torch.long).to(device)
            Y_pred = omniglot_learner(X_batch_a)

            train_loss = criterion(Y_pred, Y_batch_a)

            grad = torch.autograd.grad(train_loss, omniglot_learner.parameters(), create_graph=True)
            # pdb.set_trace()

            # gradients.append(grad)

            # 3. meta-update step
            # for i in range(metabatch_size):
                # X_batch_b, Y_batch_b = valid_data[i]
                # grad = gradients[i]

            X_batch_b = torch.tensor(X_batch_b, dtype=torch.float32).unsqueeze(1).to(device)
            Y_batch_b = torch.tensor(Y_batch_b, dtype=torch.long).to(device)

            Y_pred = omniglot_learner.forward_fast_weights(X_batch_b, grad, lr_a)
            meta_loss = criterion(Y_pred, Y_batch_b)
            meta_learning_loss += meta_loss
            # print("train_loss: {:.3f}, meta_loss: {:.3f}".format(train_loss, meta_loss))
            # pdb.set_trace()

        # 4. Backpropagation to update model's parameters
        meta_learning_loss /= num_tasks
        if iter % 10 == 0:
            print("[{}] iteration {}: meta_learning_loss = {:.5f}".format(str(datetime.now()), iter, meta_learning_loss))
            sys.stdout.flush()

        meta_learning_loss.backward()
        optimizer.step()
        optimizer.zero_grad()




    print("finished maml training")
    torch.save(omniglot_learner.state_dict(), savepath)

omniglot_maml_exp()
