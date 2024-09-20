import torch
from torch import nn
from torch.nn.functional import *


class Inception1d(nn.Module):
    """ Pytorch Implementation of Inception module for 1-dimensional data
    https://arxiv.org/pdf/1409.4842.pdf
    Inputs:
    in_channels     : input channels
    chns            : array-like, number of output channels of convolutional operations for each path.
                      The first and last path consist of one value while the other consist of 2 values.
    kernels         : array-like, kernel size of convolutional operations for each path. The first value must be 1.    
    """
    def __init__(self, in_channels, chns, kernels, act='relu', **kwargs):
        super().__init__()
        assert len(chns)==len(kernels), "Number of paths is NOT determined." 
        assert kernels[0]==1, "The first kernel_size must be 1." 
        self.in_channels = in_channels
        self.n_paths = len(chns)
        self.p = []
        self.out_channels = 0
        self.act = act
        # 1st path is a single kernel-1 convolutional layer
        if chns[0]>0:
            self.p.append(nn.Conv1d(self.in_channels, chns[0], kernel_size=kernels[0]))
            self.out_channels += chns[0]
        # middle paths include a kernel-1 convolutional layer followed by a kernel-n convolutional layer
        for i in range(1,self.n_paths-1):
            if chns[i][0] > 0 and chns[i][1] > 0:
                self.p.append(nn.Sequential(
                                  nn.Conv1d(self.in_channels, chns[i][0], kernel_size=1),
                                  nn.Conv1d(chns[i][0], chns[i][1], kernel_size=kernels[i], padding='same')))
                self.out_channels += chns[i][1]
        # last path is a kernel-n maximum pooling layer followed by a kernel-1 convolutional layer
        if chns[self.n_paths-1] > 0:
            self.p.append(nn.Sequential(
                              nn.MaxPool1d(kernel_size=kernels[self.n_paths-1], stride=1, padding=(kernels[self.n_paths-1]//2)),
                              nn.Conv1d(self.in_channels, chns[self.n_paths-1], kernel_size=1)))
            self.out_channels += chns[self.n_paths-1]
        self.p = nn.ModuleList(self.p)

    def forward(self, x):
        out = []
        for i in range(len(self.p)):
            if self.act=='linear':
                out.append(self.p[i](x))
            else:
                out.append(eval(self.act)(self.p[i](x)))
        return torch.cat(out, dim=1)
        
    def getOutChannels(self):
        return self.out_channels
        
        
class Inception2d(nn.Module):
    """ Pytorch Implementation of Inception module for 2-dimensional data
    https://arxiv.org/pdf/1409.4842.pdf
    Inputs:
    in_channels     : input channels
    chns            : array-like, number of output channels of convolutional operations for each path.
                      The first and last path consist of one value while the other consist of 2 values.
    kernels         : array-like, kernel size of convolutional operations for each path. The first value must be 1.    
    """
    def __init__(self, in_channels, chns, kernels, act='relu', **kwargs):
        super().__init__()
        assert len(chns)==len(kernels), "Number of paths is NOT determined." 
        assert kernels[0]==1, "The first kernel_size must be 1." 
        self.in_channels = in_channels
        self.n_paths = len(chns)
        self.p = []
        self.out_channels = 0
        self.act = act
        # 1st path is a single kernel-1 convolutional layer
        if chns[0]>0:
            self.p.append(nn.Conv2d(chns[0], kernel_size=kernels[0]))
            self.out_channels += chns[0]
        # middle paths include a kernel-1 convolutional layer followed by a kernel-n convolutional layer
        for i in range(1,self.n_paths-1):
            if chns[i][0] > 0 and chns[i][1] > 0:
                self.p.append(nn.Sequential(
                                  nn.Conv2d(self.in_channels, chns[i][0], kernel_size=1),
                                  nn.Conv2d(chns[i][0], chns[i][1], kernel_size=kernels[i], padding='same')))
                self.out_channels += chns[i][1]
        # last path is a kernel-n maximum pooling layer followed by a kernel-1 convolutional layer
        if chns[self.n_paths-1] > 0:
            self.p.append(nn.Sequential(
                              nn.MaxPool2d(kernel_size=kernels[self.n_paths-1], stride=1, padding=(kernels[self.n_paths-1]//2)),
                              nn.Conv2d(self.in_channels, chns[self.n_paths-1], kernel_size=1)))
            self.out_channels += chns[self.n_paths-1]
        self.p = nn.ModuleList(self.p)

    def forward(self, x):
        out = []
        for i in range(len(self.p)):
            if self.act=='linear':
                out.append(self.p[i](x))
            else:
                out.append(eval(self.act)(self.p[i](x)))
        return torch.cat(out, dim=1)
    
    def getOutChannels(self):
        return self.out_channels