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
        self.bnorm1 = nn.BatchNorm2d(num_features=64)
        self.conv2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.bnorm2 = nn.BatchNorm2d(num_features=64)
        self.conv3  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.bnorm3 = nn.BatchNorm2d(num_features=64)
        self.conv4  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.bnorm4 = nn.BatchNorm2d(num_features=64)

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
        fast_weights = self.get_fast_weights(grad, lr_a)
        conv1  = fast_weights['conv1']
        bnorm1 = fast_weights['bnorm1']
        conv2  = fast_weights['conv2']
        bnorm2 = fast_weights['bnorm2']
        conv3  = fast_weights['conv3']
        bnorm3 = fast_weights['bnorm3']
        conv4  = fast_weights['conv4']
        bnorm4 = fast_weights['bnorm4']
        fc     = fast_weights['fc']

        z1 = F.relu(bnorm1(conv1(x)))
        z2 = F.relu(bnorm2(conv2(z1)))
        z3 = F.relu(bnorm3(conv3(z2)))
        z4 = F.relu(bnorm4(conv4(z3)))

        z4 = z4.view(-1, 64)
        out = self.logsoftmax(fc(z4))
        return out

    def get_fast_weights(self, grad, lr_a):
        conv1  = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=0)
        # conv1.weight.requires_grad = False
        # conv1.bias.requires_grad   = False
        conv1.weight.data = self.conv1.weight - lr_a*grad[0]
        conv1.bias.data   = self.conv1.bias   - lr_a*grad[1]

        bnorm1 = nn.BatchNorm2d(num_features=64)
        # bnorm1.weight.requires_grad = False
        # bnorm1.bias.requires_grad   = False
        bnorm1.weight.data = self.bnorm1.weight - lr_a*grad[2]
        bnorm1.bias.data   = self.bnorm1.bias   - lr_a*grad[3]

        conv2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        # conv2.weight.requires_grad = False
        # conv2.bias.requires_grad   = False
        conv2.weight.data = self.conv2.weight - lr_a*grad[4]
        conv2.bias.data   = self.conv2.bias   - lr_a*grad[5]

        bnorm2 = nn.BatchNorm2d(num_features=64)
        # bnorm2.weight.requires_grad = False
        # bnorm2.bias.requires_grad   = False
        bnorm2.weight.data = self.bnorm2.weight - lr_a*grad[6]
        bnorm2.bias.data   = self.bnorm2.bias   - lr_a*grad[7]

        conv3  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        # conv3.weight.requires_grad = False
        # conv3.bias.requires_grad   = False
        conv3.weight.data = self.conv3.weight - lr_a*grad[8]
        conv3.bias.data   = self.conv3.bias   - lr_a*grad[9]

        bnorm3 = nn.BatchNorm2d(num_features=64)
        # bnorm3.weight.requires_grad = False
        # bnorm3.bias.requires_grad   = False
        bnorm3.weight.data = self.bnorm3.weight - lr_a*grad[10]
        bnorm3.bias.data   = self.bnorm3.bias   - lr_a*grad[11]

        conv4  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)
        # conv4.weight.requires_grad = False
        # conv4.bias.requires_grad   = False
        conv4.weight.data = self.conv4.weight - lr_a*grad[12]
        conv4.bias.data   = self.conv4.bias   - lr_a*grad[13]

        bnorm4 = nn.BatchNorm2d(num_features=64)
        # bnorm4.weight.requires_grad = False
        # bnorm4.bias.requires_grad   = False
        bnorm4.weight.data = self.bnorm4.weight - lr_a*grad[14]
        bnorm4.bias.data   = self.bnorm4.bias   - lr_a*grad[15]

        fc = nn.Linear(in_features=64, out_features=self.N_way, bias=True)
        # fc.weight.requires_grad = False
        # fc.bias.requires_grad   = False
        fc.weight.data = self.fc.weight - lr_a*grad[16]
        fc.bias.data   = self.fc.bias   - lr_a*grad[17]

        fast_weights = {
            'conv1': conv1, 'bnorm1': bnorm1,
            'conv2': conv2, 'bnorm2': bnorm2,
            'conv3': conv3, 'bnorm3': bnorm3,
            'conv4': conv4, 'bnorm4': bnorm4,
            'fc': fc
        }
        return fast_weights

# --------------------- Experiment setting --------------------- #
use_gpu = False # set to False if there is no GPU available
if use_gpu:
    device = 'cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    device = 'cpu'

omniglot_learner = Learner(N_way=5, device=device)
criterion = nn.NLLLoss(reduction='mean')

savepath = "trained_models/omniglot_model_exp0.pt"
print("savepath =", savepath)

lr_b = 0.4
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
        valid_data = []
        gradients  = []
        for task in tasks:
            X_batch_a = np.zeros((batch_size, 28, 28))
            Y_batch_a = np.zeros((batch_size))
            X_batch_b = np.zeros((batch_size, 28, 28))
            Y_batch_b = np.zeros((batch_size))

            # 2.1 sample K datapoints from Ti
            # 2.3 sample K datapoints from Ti --- for meta-update step
            for j1, char_id in enumerate(task):
                instances = np.random.randint(low=0, high=20, size=num_tasks*2)
                for j2, ins in enumerate(instances[:num_tasks]):
                    X_batch_a[j1*num_tasks+j2,:,:] = X_train[char_id,ins,:,:]
                    Y_batch_a[j1*num_tasks+j2] = j1

                for j2, ins in enumerate(instances[num_tasks:]):
                    X_batch_b[j1*num_tasks+j2,:,:] = X_train[char_id,ins,:,:]
                    Y_batch_b[j1*num_tasks+j2] = j1

            valid_data.append((X_batch_b, Y_batch_b))

            # 2.2 compute gradient
            X_batch_a = torch.tensor(X_batch_a, dtype=torch.float32).unsqueeze(1).to(device)
            Y_batch_a = torch.tensor(Y_batch_a, dtype=torch.long).to(device)
            Y_pred = omniglot_learner(X_batch_a)

            train_loss = criterion(Y_pred, Y_batch_a)

            grad = torch.autograd.grad(train_loss, omniglot_learner.parameters())
            gradients.append(grad)

        # 3. meta-update step
        meta_learning_loss = 0
        for i in range(metabatch_size):
            X_batch_b, Y_batch_b = valid_data[i]
            grad = gradients[i]

            X_batch_b = torch.tensor(X_batch_b, dtype=torch.float32).unsqueeze(1).to(device)
            Y_batch_b = torch.tensor(Y_batch_b, dtype=torch.long).to(device)

            Y_pred = omniglot_learner.forward_fast_weights(X_batch_b, grad, lr_a)
            meta_learning_loss += criterion(Y_pred, Y_batch_b)
            # print(i, meta_learning_loss)

        # 4. Backpropagation to update model's parameters
        meta_learning_loss /= num_tasks
        if iter % 10 == 0:
            print("[{}] iteration {}: meta_learning_loss = {:.3e}".format(str(datetime.now()), iter, meta_learning_loss))
            sys.stdout.flush()

        meta_learning_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    print("finished maml training")
    torch.save(reg_learner.state_dict(), savepath)

omniglot_maml_exp()
