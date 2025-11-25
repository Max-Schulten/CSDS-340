#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 23:03:29 2025

@author: maximilianschulten
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Config
train_end_index = 3511
sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_suffix = '_train_1.csv'

#%% Data Load

 # Load labels and training sensor data into 3-D array
labels = np.loadtxt('Train_1/labels_train_1.csv', dtype='int')
data_slice_0 = np.loadtxt('Train_1/' + sensor_names[0] + '_train_1.csv',
                          delimiter=',')
data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
                 len(sensor_names)))
data[:, :, 0] = data_slice_0
del data_slice_0
for sensor_index in range(1, len(sensor_names)):
    data[:, :, sensor_index] = np.loadtxt(
        'Train_1/' + sensor_names[sensor_index] + '_train_1.csv', delimiter=',')

# Split into training and test by row index. Do not use a random split as
# rows are not independent!
train_data = data[:train_end_index+1, :, :]
train_labels = labels[:train_end_index+1]
test_data = data[train_end_index+1:, :, :]
test_labels = labels[train_end_index+1:]

n_classes = np.unique(train_labels).size

class HARDataset(Dataset):
    def __init__(self, X, y):
        X_std = (X- np.mean(X)) / np.std(X)
        y_shifted = y-1
        self.X = torch.from_numpy(X_std).float()
        self.y = torch.from_numpy(y_shifted).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_ds = HARDataset(train_data, train_labels)
test_ds = HARDataset(test_data, test_labels)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
#%% MLP on Flattened Input Data
class HARMLP(nn.Module):
    def __init__(self, input_dim=60*6, n_classes=n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Dropout(0.75),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        return self.net(x)
    
    def predict(self, dl, batch_size=64, device="cpu"):
        self.eval()
        all_preds = []
    
        with torch.no_grad():
            for Xb, _ in dl:
                Xb = Xb.to(device)
                logits = self.forward(Xb)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
    
        return torch.cat(all_preds)
        
model = HARMLP(input_dim=360, n_classes=n_classes)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
    
#%% Training
EPOCHS = 1000
loss_hist =[0]*EPOCHS

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, total = 0.0, 0
    all_preds = []
    all_targets = []

    for X_epoch, y_epoch in loader:
        X_epoch, y_epoch = X_epoch.to(device), y_epoch.to(device)
        
        optimizer.zero_grad()
        out = model(X_epoch)
        loss = criterion(out, y_epoch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_epoch.size(0)
        preds = out.argmax(dim=1)

        all_preds.append(preds.detach().cpu())
        all_targets.append(y_epoch.detach().cpu())

        total += y_epoch.size(0)
    
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    micro = f1_score(y_true, y_pred, average='micro')
    macro = f1_score(y_true, y_pred, average='macro')

    return running_loss / total, micro, macro


for epoch in range(EPOCHS):
    loss, micro, macro = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
    loss_hist[epoch] = loss
    print(f"Epoch: {epoch}\n",
          f"\tLoss: {loss}\n",
          f"\tMicro f1: {micro}\n",
          f"\tMacro f1: {macro}")

#%% Testing 

model.eval()
preds = torch.argmax(model(test_ds.X), dim=1)
micro = f1_score(test_ds.y, preds, average='micro')
macro = f1_score(test_ds.y, preds, average='macro')
print(
      f"\tMicro f1: {micro}\n",
      f"\tMacro f1: {macro}")
