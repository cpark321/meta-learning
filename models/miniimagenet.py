import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnerConv(nn.Module):
    def __init__(self, N_way, device):
        super(LearnerConv, self).__init__()
        self.device = device
        self.N_way  = N_way
        self.conv1  = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bnorm1 = nn.BatchNorm2d(num_features=32, track_running_stats=False)

        self.conv2  = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bnorm2 = nn.BatchNorm2d(num_features=32, track_running_stats=False)

        self.conv3  = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bnorm3 = nn.BatchNorm2d(num_features=32, track_running_stats=False)

        self.conv4  = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bnorm4 = nn.BatchNorm2d(num_features=32, track_running_stats=False)

        self.fc = nn.Linear(in_features=800, out_features=N_way, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.to(device)

    def copy_model_weights(self):
        fast_weights = []
        for pr in self.parameters():
            fast_weights.append(pr.clone())
        return fast_weights

    def update_fast_grad(self, fast_params, grad, lr_a):
        if len(fast_params) != len(grad): raise ValueError("fast grad update error")
        num = len(grad)
        updated_fast_params = [None for _ in range(num)]
        for i in range(num):
            updated_fast_params[i] = fast_params[i] - lr_a * grad[i]
        return updated_fast_params

    def forward(self, x):
        z1 = F.max_pool2d(F.relu(self.bnorm1(self.conv1(x))), kernel_size=2)
        z2 = F.max_pool2d(F.relu(self.bnorm2(self.conv2(z1))), kernel_size=2)
        z3 = F.max_pool2d(F.relu(self.bnorm3(self.conv3(z2))), kernel_size=2)
        z4 = F.relu(self.bnorm4(self.conv4(z3)))
        z4 = F.max_pool2d(z4, kernel_size=2, stride=1)

        z4 = z4.view(-1, 800)
        out = self.logsoftmax(self.fc(z4))

        return out

    def forward_fast_weights(self, x, fast_weights):
        conv1_w  = fast_weights[0]
        conv1_b  = fast_weights[1]
        bnorm1_w = fast_weights[2]
        bnorm1_b = fast_weights[3]

        conv2_w  = fast_weights[4]
        conv2_b  = fast_weights[5]
        bnorm2_w = fast_weights[6]
        bnorm2_b = fast_weights[7]

        conv3_w  = fast_weights[8]
        conv3_b  = fast_weights[9]
        bnorm3_w = fast_weights[10]
        bnorm3_b = fast_weights[11]

        conv4_w  = fast_weights[12]
        conv4_b  = fast_weights[13]
        bnorm4_w = fast_weights[14]
        bnorm4_b = fast_weights[15]

        fc_w     = fast_weights[16]
        fc_b     = fast_weights[17]

        z1 = F.conv2d(x, conv1_w, conv1_b, stride=1, padding=0)
        z1 = F.batch_norm(z1, running_mean=self.bnorm1.running_mean, running_var=self.bnorm1.running_var, weight=bnorm1_w, bias=bnorm1_b, training=True) # how about training=True??
        z1 = F.relu(z1)
        z1 = F.max_pool2d(z1, kernel_size=2, stride=2)

        z2 = F.conv2d(z1, conv2_w, conv2_b, stride=1, padding=0)
        z2 = F.batch_norm(z2, running_mean=self.bnorm2.running_mean, running_var=self.bnorm2.running_var, weight=bnorm2_w, bias=bnorm2_b, training=True) # how about training=True??
        z2 = F.relu(z2)
        z2 = F.max_pool2d(z2, kernel_size=2, stride=2)

        z3 = F.conv2d(z2, conv3_w, conv3_b, stride=1, padding=0)
        z3 = F.batch_norm(z3, running_mean=self.bnorm3.running_mean, running_var=self.bnorm3.running_var, weight=bnorm3_w, bias=bnorm3_b, training=True) # how about training=True??
        z3 = F.relu(z3)
        z3 = F.max_pool2d(z3, kernel_size=2, stride=2)

        z4 = F.conv2d(z3, conv4_w, conv4_b, stride=1, padding=0)
        z4 = F.batch_norm(z4, running_mean=self.bnorm4.running_mean, running_var=self.bnorm4.running_var, weight=bnorm4_w, bias=bnorm4_b, training=True) # how about training=True??
        z4 = F.relu(z4)
        z4 = F.max_pool2d(z4, kernel_size=2, stride=1)

        z4 = z4.view(-1, 800) # batch_size x 32*5*5
        z4 = torch.matmul(z4, fc_w.T) + fc_b
        out = F.log_softmax(z4, dim=-1)

        return out
