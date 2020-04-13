import torch
import torch.nn as nn
import numpy as np


from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from datetime import datetime
import sys

from utils import MVTecMetaDataset, MVTecDataset
from models import MVTecLearner
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--target', required=True, help='target class')
# parser.add_argument('-n', '--n_way', type=int, required=True)
parser.add_argument('-k', '--k_shot', type=int, required=True, default=None)
parser.add_argument('-c', '--no_cuda', required=False, default=None)
parser.add_argument('--no_iter', default= 50000, type= int, required=False, help='number of epochs')
parser.add_argument('--no_grad', default= 1, type= int, required=False, help='number of gradient updates')
parser.add_argument('--lr', default= 0.001, type=float , required=False, help='learning rate')

args = parser.parse_args()

num_tasks = 2   # two classes : normal / abnormal

target_class = args.target
target_item = target_class

num_points = args.k_shot
num_iterations = args.no_iter

data_dir = './data/data'
save_path = os.path.join('./mvtec_saves/', f'{num_tasks}-way-{num_points}-shot')

if not os.path.exists(save_path):
    os.makedirs(save_path)

device_type='cuda'

if args.no_cuda is not None:
    device_type = 'cuda:'+args.no_cuda

device = torch.device(device_type if torch.cuda.is_available() else 'cpu')

def MetaTrainDataset(metaTargetDicts):
    metaNormalImgs ={}
    metaAbnormalImgs = {}

    for i in metaTargetDicts:
        normal_list_dir = [os.path.join(data_dir, metaTargetDicts[i], 'train', 'good'), os.path.join(data_dir, metaTargetDicts[i], 'test', 'good')]

        test_dir = os.path.join(data_dir, metaTargetDicts[i], 'test')
        test_subfolders = next(os.walk(test_dir))[1]

        abnormal_list_dir=[]

        for item in test_subfolders:
            if item != 'good':
                abnormal_list_dir.append(os.path.join(data_dir, metaTargetDicts[i], 'test', item))

        metaNormalImgs[i] = MVTecMetaDataset(normal_list_dir)
        metaAbnormalImgs[i] = MVTecMetaDataset(abnormal_list_dir)
        
        
        
    return metaNormalImgs, metaAbnormalImgs

def TestDataset(target_class):    
    normal_list_dir = [os.path.join(data_dir, target_class, 'train', 'good'), os.path.join(data_dir, target_class, 'test', 'good')]

    test_dir = os.path.join(data_dir, target_class, 'test')
    test_subfolders = next(os.walk(test_dir))[1]

    abnormal_list_dir=[]

    for item in test_subfolders:
        if item != 'good':
            abnormal_list_dir.append(os.path.join(data_dir, target_class, 'test', item))

    dataset = MVTecDataset(normal_list_dir, abnormal_list_dir)
    
    val_num = int(len(dataset)*0.15)
    test_num = int(len(dataset)*0.15)
    train_num = len(dataset) - val_num - test_num

    train_dataset, valid_dataset, test_dataset =random_split(dataset,[train_num, val_num, test_num])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, valid_loader, test_loader


metaTargets = next(os.walk(data_dir))[1]
metaTargetDicts = {}
count=0
for target in metaTargets:
    if target == target_item:
        continue
    metaTargetDicts[count] = target
    count+=1
    
metaNormalImgs, metaAbnormalImgs = MetaTrainDataset(metaTargetDicts)
# train_loader, valid_loader, test_loader = TestDataset(target_item)

batch_size      = num_tasks*num_points
metabatch_size  = 14  # 14 other tasks --- the number of tasks sampled per meta-update
                     #          --- each task is an N-way, K-shot classification problem

lr_a = args.lr
num_grad_update = args.no_grad

print("N={}".format(num_tasks))
print("K={}".format(num_points))
print("metabatch_size={}".format(metabatch_size))
print("lr_a={}".format(lr_a))
print("num_grad_update={}".format(num_grad_update))

mvtec_learner = MVTecLearner(device=device)

lr_b = 1e-4
print("lr_beta = {:.2e}".format(lr_b))

def criterion(x, y):
    remove_zero_losses = (x!=y)
    x = x[remove_zero_losses]
    y = y[remove_zero_losses]
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()

optimizer = torch.optim.Adam(mvtec_learner.parameters(), lr=lr_b, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
optimizer.zero_grad()

min_loss = np.inf

for epoch in range(num_iterations):

    # 2. for each 14 other tasks, Ti
    meta_learning_loss = 0
    labels = [0,1]
    
    for task in range(metabatch_size):

        # copy current model weights to fast_weights
        fast_weights = mvtec_learner.copy_model_weights()

        # 2.1 sample K datapoints from Ti
        normal_sampler = DataLoader(metaNormalImgs[task], batch_size=num_points, shuffle=True)
        normal_imgs, _ = next(iter(normal_sampler))
        abnormal_sampler = DataLoader(metaAbnormalImgs[task], batch_size=num_points, shuffle=True)
        abnormal_imgs, _ = next(iter(abnormal_sampler))    
        label_list = [[item] for item in labels for i in range(num_points)]

        X_batch_a = torch.cat((normal_imgs, abnormal_imgs), dim=0).to(device)
        Y_batch_a = torch.tensor(label_list, dtype=torch.float).to(device)


          # 2.2 compute gradient (multiple steps)
        for grad_update_iter in range(num_grad_update):
            Y_pred = mvtec_learner.forward_fast_weights(X_batch_a, fast_weights)
            train_loss = criterion(Y_pred, Y_batch_a)
            grad = torch.autograd.grad(train_loss, fast_weights, create_graph=True)
            fast_weights = mvtec_learner.update_fast_grad(fast_weights, grad, lr_a)  

        # 2.3 sample K datapoints from Ti --- for meta-update step    
        normal_imgs, _ = next(iter(normal_sampler))
        abnormal_imgs, _ = next(iter(abnormal_sampler))    
        label_list = [[item] for item in labels for i in range(num_points)]

        X_batch_b = torch.cat((normal_imgs, abnormal_imgs), dim=0).to(device)
        Y_batch_b = torch.tensor(label_list, dtype=torch.float).to(device)

        # 3. meta-update step
        Y_pred = mvtec_learner.forward_fast_weights(X_batch_b, fast_weights)

        meta_loss = criterion(Y_pred, Y_batch_b)
        meta_learning_loss += meta_loss

    # 4. Backpropagation to update model's parameters    
    meta_learning_loss /= metabatch_size

    meta_learning_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 100==0:
        with open(os.path.join(save_path, f'training-log-target-{target_class}-lr{lr_a}.txt'), 'a') as f:
            f.write("[{}] iter {}: meta_learning_loss = {:.3e}\n".format(str(datetime.now()), epoch, meta_learning_loss))
        if meta_learning_loss.item() < min_loss:
            min_loss = meta_learning_loss.item()
            savepath = os.path.join(save_path, "mvtec_target{}_n{}_k{}_lr{}_final.pth".format(target_class, num_tasks, num_points, lr_a))
            torch.save(mvtec_learner.state_dict(), savepath)

print("finished maml training")


