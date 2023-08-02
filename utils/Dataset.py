import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

def get_target_label(label):
    zero_index = np.where(label == 1)
    zero_index = np.array(zero_index).reshape(len(zero_index[0]))
    # random.seed(3)
    target_index = random.choice(zero_index)
    queryL = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    queryL[target_index] = 1
    re_queryL =label-queryL
    return queryL,re_queryL

class HashingDataset(Dataset):
    def __init__(self,
                 target=False,
                 txt_path='/mm/data/NUS-WIDE',
                 data_path='/dataset/NUS-WIDE',
                 img_filename='train_img.txt',
                 label_filename='train_label.txt',
                 transform=transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor()
                 ])):
        self.target = target
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(txt_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(txt_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).float()
        if self.target:
            label,re_label = get_target_label(label)
            return img, label, re_label, index
        else:
            return img, label, index

    def __len__(self):
        return len(self.img_filename)

class HashingDataset_part(Dataset):
    def __init__(self,
                 take_index,
                 txt_path='/data/NUS-WIDE',
                 data_path='/dataset/NUS-WIDE',
                 img_filename='train_img.txt',
                 label_filename='train_label.txt',
                 target=False,
                 transform=transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor()
                 ])):
        self.target = target
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(txt_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        self.img_filename_ = [self.img_filename[i] for i in take_index]
        label_filepath = os.path.join(txt_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)
        self.label_ = [self.label[i] for i in take_index]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename_[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label_[index]).float()
        if self.target:
            label = get_target_label(label)
        return img, label, index

    def __len__(self):
        return len(self.img_filename_)

def load_model(path):
    model = torch.load(path)
    model = model.cuda()
    model.eval()
    return model

def load_label(data_dir):
    label = np.loadtxt(data_dir, dtype=np.int64)
    return torch.from_numpy(label).float()

def generate_hash_code(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B

def generate_code_label(model, data_loader, num_data, bit, num_class):
    B = torch.zeros([num_data, bit]).cuda()
    L = torch.zeros(num_data, num_class).cuda()
    for iter, data in enumerate(data_loader, 0):
        data_input, data_label, data_ind = data
        output = model(data_input.cuda())
        B[data_ind, :] = torch.sign(output.data)
        L[data_ind, :] = data_label.cuda()
    return B, L