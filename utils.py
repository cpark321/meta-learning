import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


def generate_file_list(list_dir):
    extensions = ('.png', '.PNG')
    file_list = []
    for path in list_dir:
        for filename in os.listdir(path):
            if filename.endswith(extensions):
                fullPath = os.path.join(path, filename)
                file_list.append(fullPath)
    return file_list


def evaluate_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for (img, label) in dataloader:
            img = img.to(device)
            label = label.to(device, dtype=torch.float)
            label = label.view(-1, 1)
            outputs = model(img)
            predicted = (outputs.data > 0.5).float().to(device)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return correct / total


class MVTecDataset(Dataset):
    def __init__(self, normal_data_dir, abnormal_data_dir):
        super(MVTecDataset, self).__init__()
        self.normal_data = generate_file_list(normal_data_dir)
        self.abnormal_data = generate_file_list(abnormal_data_dir)
        self.transform = transforms.Compose(
            [transforms.Resize([300, 300]), transforms.RandomCrop(256), transforms.RandomHorizontalFlip(), \
             transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.normal_label = [1 for x in range(len(self.normal_data))]
        self.abnormal_label = [0 for x in range(len(self.abnormal_data))]

        self.data_list = self.normal_data + self.abnormal_data * int(len(self.normal_label) / len(self.abnormal_label))
        self.label_list = self.normal_label + self.abnormal_label * int(
            len(self.normal_label) / len(self.abnormal_label))

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_name = self.data_list[index]
        img = Image.open(image_name)

        if len(np.array(img).shape) == 2:
            img = img.convert(mode="RGB")

        img = self.transform(img)
        label = self.label_list[index]
        return (img, label)


class MVTecMetaDataset(Dataset):
    def __init__(self, data_dir):
        super(MVTecMetaDataset, self).__init__()
        self.data_list = generate_file_list(data_dir)
        self.transform = transforms.Compose(
            [transforms.Resize([300, 300]), transforms.RandomCrop(256), transforms.RandomHorizontalFlip(), \
             transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.label_list = [0 for x in range(len(self.data_list))]
        
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_name = self.data_list[index]
        img = Image.open(image_name)

        if len(np.array(img).shape) == 2:
            img = img.convert(mode="RGB")

        img = self.transform(img)
        label = self.label_list[index]
        return (img, label)    
    
    
    
    
    
    
