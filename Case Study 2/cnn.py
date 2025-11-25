#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 17:20:56 2025

@author: maximilianschulten
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Config
train_end_index = 3511
sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_suffix = '_train_1.csv'
torch.manual_seed(531)

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

train_mean = train_data.mean(axis=(0, 1), keepdims=True)
train_std  = train_data.std(axis=(0, 1), keepdims=True)

n_classes = np.unique(train_labels).size

class HARDataset(Dataset):
    def __init__(self, X, y, mean, std, augment=False, noise_std=0.02):

        X_std = (X - mean) / (std + 1e-8)
        y_shifted = y-1
        self.X = torch.from_numpy(X_std).float()
        self.y = torch.from_numpy(y_shifted).long()
        
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.augment:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x, self.y[idx]
    
train_ds = HARDataset(train_data, train_labels, train_mean, train_std, augment=True, noise_std=0.01)
test_ds = HARDataset(test_data, test_labels, train_mean, train_std, augment=False)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
#%% CNN on Flattened Input Data
class HARConvNet(nn.Module):
    def __init__(self, n_channels=6, n_classes=4):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.15),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            #nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        if x.shape[1] != 6:
            x = x.permute(0, 2, 1)

        x = self.features(x)
        x = self.classifier(x)
        return x
        
model = HARConvNet(n_channels=6, n_classes=n_classes)
model.to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5,
)

#%% Training
EPOCHS = 75
loss_hist = [0]*EPOCHS
train_micro_hist = [0]*EPOCHS
train_macro_hist = [0]*EPOCHS
test_micro_hist = [0]*EPOCHS
test_macro_hist = [0]*EPOCHS


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

    # store training metrics
    loss_hist[epoch] = loss
    train_micro_hist[epoch] = micro
    train_macro_hist[epoch] = macro

    # eval on test set and store
    model.eval()
    with torch.no_grad():
        logits = model(test_ds.X.to(device))
        preds = torch.argmax(logits, dim=1).cpu()

    test_micro = f1_score(test_ds.y, preds, average='micro')
    test_macro = f1_score(test_ds.y, preds, average='macro')

    test_micro_hist[epoch] = test_micro
    test_macro_hist[epoch] = test_macro

    print(
        f"Epoch {epoch}\n"
        f"  Train loss:      {loss:.4f}\n"
        f"  Train micro F1:  {micro:.4f}\n"
        f"  Train macro F1:  {macro:.4f}\n"
        f"  Test micro F1:   {test_micro:.4f}\n"
        f"  Test macro F1:   {test_macro:.4f}"
    )

#%% Testing 

plt.clf()
plt.plot(train_micro_hist, label="Train Micro F1")
plt.plot(test_micro_hist, label="Test Micro F1")
plt.plot(train_macro_hist, label="Train Macro F1")
plt.plot(test_macro_hist, label="Test Macro F1")
#plt.plot(loss_hist, label="TCross-Entropy Loss")
plt.xlabel("Training Epoch")
plt.ylabel("Score")
plt.ylim(0,1.1)
plt.title("CNN Training Progression")
plt.legend(loc="lower right")
plt.show()

plt.savefig("cnn_pre_sub.png", dpi=400)

