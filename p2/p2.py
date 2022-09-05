# %%
from ast import Num
import torch as t
import torch.nn.functional as F
from tqdm import tqdm
from tqdm import trange
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import re
import os
import numpy as np
# %%
config = {
    'train_meta_path': 'H:\\我的雲端硬碟\\wyshih629.cs01g@g2.nctu.edu.tw 2022-07-17 10 37\\wyshih_Research\\Courses\\NN\\HW2\\libriphone\\train_split.txt',
    'train_label_path': 'H:\\我的雲端硬碟\\wyshih629.cs01g@g2.nctu.edu.tw 2022-07-17 10 37\\wyshih_Research\\Courses\\NN\\HW2\\libriphone\\train_labels.txt',
    'test_meta_path': 'H:\\我的雲端硬碟\\wyshih629.cs01g@g2.nctu.edu.tw 2022-07-17 10 37\\wyshih_Research\\Courses\\NN\\HW2\\libriphone\\test_split.txt',
    'files_path': 'H:\\我的雲端硬碟\\wyshih629.cs01g@g2.nctu.edu.tw 2022-07-17 10 37\\wyshih_Research\\Courses\\NN\\HW2\\libriphone\\feat'
}
# %%


class VoiceSingleData(Dataset):
    def __init__(self, filespath, metapath, labelpath=None):

        self.filespath = filespath
        if labelpath != None:
            self.train = True
        else:
            self.train = False
        with open(metapath, 'r') as fd:
            self.meta = [i.strip() for i in fd.readlines()]
        if labelpath != None:
            self.ally = {}
            with open(labelpath, 'r') as ld:
                content = ld.readlines()
                for i in content:
                    i = i.strip().split()
                    self.ally.setdefault(i[0], list(map(int, i[1:])))

    def __getitem__(self, idx):
        file = self.meta[idx]
        filepath = os.path.join(
            self.filespath, self.train and 'train' or 'test', file+'.pt')
        if self.train:
            return t.load(filepath), F.one_hot(t.tensor(self.ally[file]), num_classes=41)
        else:
            return t.load(filepath)

    def __len__(self):
        return len(self.meta)


class VoiceData(Dataset):
    def __init__(self, alldata, extends=3, extende=3):
        self.feature_len = extends+extende+1
        self.alldata = alldata
        self.extends = extends
        self.extende = extende
        self.ind = 0
        self.x = t.empty(3000000, 39*self.feature_len)
        if self.alldata.train == True:
            self.y = t.empty(1, 41)
        self.builddata()

    def concatedata(self, single):
        temp = t.zeros(1, 39)
        for j in range(single.shape[0]):
            # print(self.alldata[i][0][j])
            if j < self.extends:
                data = temp.repeat(1, self.extends-j)
                data = t.cat(
                    [data, t.flatten(single[:j+1]).reshape(1, -1)], dim=1)
            else:
                data = t.flatten(single[j-self.extends:j+1]).reshape(1, -1)
            if j+self.extende >= single.shape[0]:
                data = t.cat(
                    [data, t.flatten(single[j+1:]).reshape(1, -1)], dim=1)
                data = t.cat(
                    [data, temp.repeat(1, j+self.extende+1-single.shape[0])], dim=1)
            else:
                data = t.cat(
                    [data, t.flatten(single[j+1:j+self.extende+1]).view(1, -1)], dim=1)
            #print(self.extends-j, j+self.extende+1-single.shape[0], single[:j+1].shape, single[j+1:j+self.extende+1].shape, data.shape)
            self.x[self.ind] = data.detach()
            self.ind += 1

    def builddata(self):
        for i in trange(len(self.alldata)):
            if self.alldata.train == True:
                self.concatedata(self.alldata[i][0])
                self.y = t.cat([self.y, self.alldata[i][1]])
                # break
            else:
                self.concatedata(self.alldata[i])
                # break
        self.x = self.x[:self.ind].detach()
        if self.alldata.train == True:
            self.y = self.y[1:].detach()


# %%
traindata = VoiceSingleData(
    config['files_path'], config['train_meta_path'], config['train_label_path'])
testdata = VoiceSingleData(config['files_path'], config['test_meta_path'])
# trainloader = DataLoader(traindata, batch_size=50,
#                         shuffle=True, pin_memory=True)
# testloader = DataLoader(testdata, batch_size=50,
#                        shuffle=False, pin_memory=True)
x = VoiceData(traindata)
# %%
