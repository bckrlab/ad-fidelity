import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "models"

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, *args, **kwargs)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        # self.dropout = nn.Dropout3d(p=0.1) # maybe channel dropout is to much?
        # self.dropout = nn.Dropout(p=0.3)
        self.max_pool3d = nn.MaxPool3d(2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv3d(x)
        # batchnorm learns normalization parameters during training
        x = self.batchnorm(x)
        x = self.max_pool3d(x)
        x = self.relu(x)
        # x = self.dropout(x)
        return x

class AdniNet(nn.Module):
    """Net Architecture like in Notebook 2"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb1 = ConvBlock(1, 5, 3, padding="same")
        self.cb2 = ConvBlock(5, 5, 3, padding="same")
        self.cb3 = ConvBlock(5, 5, 3, padding="same")
        self.linear1 = nn.Linear(10800, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        # in-dim: Batch, 1, 100, 120, 100
        # x = x.view(-1, 1, 100, 120, 100) # hinzufügen der Channel-Dimension
        # each convolutional block
        # halves the size of each dimension, rounded down (maxpool3d)
        x = self.cb1(x)
        # 5 (channels) * f(100/2) = 50 * f(120/2) = 60 * f(100/2) = 50
        x = self.cb2(x)
        # 5 (channels) * f(50/2) = 25 * f(60/2) = 30 * f(50/2) = 25
        x = self.cb3(x)
        # 5 (channels) * f(25/2) = 12 * f(30/2) = 15 * f(25/2) = 12
        x = self.dropout(x.view(-1, 10800)) # flatten
        x = self.dropout(self.linear1(x))
        x = self.dropout(self.linear2(x))
        x = self.linear3(x)
        # return F.log_softmax(x, dim=1)
        return x 

class Conv4Net(nn.Module):
    """4 Convolution Blocks to minimize Parameter number. """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb1 = ConvBlock(1, 5, 3)
        self.cb2 = ConvBlock(5, 5, 3)
        self.cb3 = ConvBlock(5, 5, 3)
        self.cb4 = ConvBlock(5, 5, 3)
        #self.dropout3d = nn.Dropout3d(p=0.2)
        
        self.linear1 = nn.Linear(875, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        # in-dim: 121, 145, 121
        x = x.view(-1, 1, 121, 145, 121) # hinzufügen der Channel-Dimension
        # each convolutional block:
        # 1. makes each dimension 2 smaller (kernel size 3)
        # 2. halves the size of each dimension, rounded down (maxpool3d)
        x = self.cb1(x)
        # 5 (channels) * f(119/2) = 59 * f(143/2) = 71 * f(119/2) = 59
        x = self.cb2(x)
        # 5 (channels) * f(57/2) = 28 * f(69/2) = 34 * f(57/2) = 28
        x = self.cb3(x)
        # 5 (channels) * f(26/2) = 13 * f(32/2) = 16 * f(26/2) = 13
        x = self.cb4(x)
        # 5 (channels) * f(11/2) = 5 * f(14/2) = 7 * f(11/2) = 5
        x = self.dropout(x.view(-1, 875)) # flatten
        x = self.dropout(self.linear1(x))
        x = self.dropout(self.linear2(x))
        x = self.linear3(x)
        # return F.log_softmax(x, dim=1)
        return x