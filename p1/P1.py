# %%
import torch as T
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
import numpy as np


class CovidData(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        if self.y != None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return self.x.shape[0]


class Model(nn.Module):
    def __init__(self, ind):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(ind, 3),
                                    nn.ReLU(),

                                    # nn.BatchNorm2d(2),
                                    #nn.Linear(3, 2),
                                    nn.Linear(3, 1))

    def forward(self, x):
        outp = self.layers(x)
        return outp


def training(trainloader, validloader, idim, device, L2, momentum, epoch=3500, batch_size=50, lr=1e-6, early_stop=200):
    model = Model(idim)
    lossf = nn.MSELoss(reduction='mean')
    optimizer = T.optim.SGD(model.parameters(), lr=lr,
                            weight_decay=L2, momentum=momentum)
    scheduler = T.optim.lr_scheduler.MultiStepLR(
        optimizer, np.linspace(1, epoch, 2), gamma=0.1)
    step = 0
    count = 0
    best = np.inf
    for i in trange(epoch):
        model.train()
        loss_record = []
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = lossf(output, y)
            loss_record.append(loss.detach().item())
            loss.backward()
            optimizer.step()
        print("Trainloss", sum(loss_record)/len(loss_record))
        model.eval()
        loss_record = []
        for x, y in validloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = lossf(pred, y)
            loss_record.append(loss.item())
        validloss = sum(loss_record)/len(loss_record)
        if validloss < best:
            best = validloss
            count = 0
            T.save(model.state_dict(),
                   './best_model.pkl')
            print("Save model at loss {0}".format(validloss))
        else:
            count += 1
        if count >= early_stop:
            break
        print("Valid Error", validloss)
        step += 1
    # return model


def feature(df):
    drop_features = ['cli', 'ili', 'depressed', 'worried_finances', 'anxious']
    pattern = re.compile(
        # r'(cli|ili|depressed|worried_finances|anxious|hh_cmnty_cli|nohh_cmnty_cli|large_event|public_transit)')
        r"(tested_positive)")
    for i in df.columns:
        if pattern.search(i) == None:
            df.drop(columns=i, inplace=True)
    df.drop(columns=['tested_positive'], inplace=True)
    df.drop(columns=['tested_positive.1'], inplace=True)

    return df


config = {'test_size': 0.3, 'L2': 0, 'momentum': 0.8,
          'epoch': 100, 'batch_size': 200, 'lr': 1e-6}

device = 'cuda' if T.cuda.is_available() else 'cpu'
traindf = pd.read_csv(
    "H:\\我的雲端硬碟\\wyshih629.cs01g@g2.nctu.edu.tw 2022-07-17 10 37\\wyshih_Research\\Courses\\NN\\HW1\\covid.train.csv")
traindf.drop(columns=traindf.columns[:39], inplace=True)
traindf = feature(traindf)

traindata = T.FloatTensor(traindf.values)
x_train, x_valid, y_train, y_valid = train_test_split(
    traindata[:, :-1], traindata[:, -1], test_size=config['test_size'])
train = CovidData(x_train[:10], y_train[:10])
valid = CovidData(x_valid[:10], y_valid[:10])
train_loader = DataLoader(
    train, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(
    valid, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
# model = training(train_loader, valid_loader, x_train.shape[1], device, config['L2'], config['momentum'], epoch=config['epoch'],
#                 batch_size=config['batch_size'], lr=config['lr'])

# %%
testdf = pd.read_csv(
    "H:\\我的雲端硬碟\\wyshih629.cs01g@g2.nctu.edu.tw 2022-07-17 10 37\\wyshih_Research\\Courses\\NN\\HW1\\covid.test.csv")
testdf.drop(columns=testdf.columns[:39], inplace=True)
testdf = feature(testdf)
testdata = T.FloatTensor(testdf.values)
test = CovidData(testdata)
model = Model(x_train.shape[1]).to(device)
model.load_state_dict(T.load('./best_model.pkl'))
model.eval()
testloader = DataLoader(test, batch_size=50, shuffle=False)
result = []
for x in testloader:
    x = x.to(device)
    pred = model(x)
    result.extend(pred.detach())
repd = pd.DataFrame([(i, d.item()) for i, d in enumerate(
    result)], columns=['id', 'tested_positive'])
repd.to_csv('pred.csv', index=False)
# %%
