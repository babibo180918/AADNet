import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseNet import *
import matplotlib.pyplot as plt

from .inception import Inception1d
from .loss import CorrelationLoss

class EEGNetAAD(nn.Module):
    """ Pytorch re-implementation of CNN model for envelope reconstruction method
    https://iopscience.iop.org/article/10.1088/1741-2552/ac7976
    The body of the model is mainly copied from the original implementation at https://github.com/Mike-boop/mldecoders
    Inputs:
    config      : configuration dictionary
    L           : length of input EEG in samples
    n_streams   : number of input audio streams
    sr          : data sampling rate
    channels    : list of EEG channels
    """       
    def __init__(self, config:dict, L, n_streams, sr, channels):
        super().__init__()
        self.sr = sr
        self.channels = channels
        self.L = L
        self.n_streams = n_streams
        
        self.n_chns = len(self.channels)           
        self.context = int(config['context']*self.sr) # input_length
        self.F1 = config['F1']
        self.D = config['D']
        self.F2 = config['F2']
        self.dropout = config['dropout']

        self.conv1_kernel=3
        self.conv3_kernel=3
        self.temporalPool1 = 2
        self.temporalPool2 = 5

        # input shape is [1, C, T]
        
        self.conv1 = torch.nn.Conv2d(1, self.F1, (1, self.conv1_kernel), padding='same')
        self.conv2 = torch.nn.Conv2d(self.F1, self.F1*self.D, (self.n_chns, 1), padding='valid', groups=self.F1)
        self.conv3 = torch.nn.Conv2d(self.F1*self.D, self.F1*self.D, (1, self.conv1_kernel), padding='same', groups=self.F1*self.D)
        self.conv4 = torch.nn.Conv2d(self.F1*self.D, self.F2, (1,1))

        self.pool1 = torch.nn.AvgPool2d((1, self.temporalPool1))
        self.pool2 = torch.nn.AvgPool2d((1, self.temporalPool2))
        self.linear = torch.nn.Linear(self.F2*(self.context//(self.temporalPool1*self.temporalPool2)), 1)

        self.batchnorm1 = torch.nn.BatchNorm2d(self.F1)
        self.batchnorm2 = torch.nn.BatchNorm2d(self.F1*self.D)
        self.batchnorm3 = torch.nn.BatchNorm2d(self.F2)

        self.dropout1 = torch.nn.Dropout2d(self.dropout)
        self.dropout2 = torch.nn.Dropout2d(self.dropout)

        self.activation1 = torch.nn.ELU()
        self.activation2 = torch.nn.ELU()
        self.activation3 = torch.nn.ELU()

    def __str__(self):
        return 'cnn'
    
    def forward(self, eeg, env, y=None):
        (batch_size,c,eeg_L) = eeg.shape
        (_,n_streams,_) = env.shape
        assert self.n_streams==n_streams, "Number of streams of the input data is mismatched."
        L = eeg_L - self.context + 1
        out = torch.zeros(batch_size,L).to(eeg.device)
        for i in range(L):
            x = eeg[...,i:i+self.context]
            #x shape = [batch, C, T]
            x = x.unsqueeze(1)
            x = self.conv1(x)
            x = self.batchnorm1(x)

            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.activation1(x)
            x = self.pool1(x)
            x = self.dropout1(x)

            #shape is now [batch, DxF1, 1, T//TPool1]
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.batchnorm3(x)
            x = self.activation2(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            x = torch.flatten(x, start_dim = 1) # shape is now [batch, F2*T//(TPool1*TPool2)]
            x = self.linear(x)
            out[...,i] = x.flatten()
        return out        

    def initialize(self, state=None):
        for name, param in self.named_parameters():
            if param.requires_grad and 'weight' in name:
                if 'batchnorm' in name:
                    continue
                else:
                    nn.init.xavier_uniform_(param)
        if state is not None:
            self.load_state_dict(state)
                        
class AADNet(nn.Module):
    """ Pytorch implementation of AADNet model for directly identifying auditory attended speaker
    Inputs:
    config      : configuration dictionary
    L           : length of input EEG in samples
    n_streams   : number of input audio streams
    sr          : data sampling rate
    channels    : list of EEG channels
    """  
    def __init__(self, config:dict, L, n_streams, sr, channels):
        super().__init__()
        self.sr = sr
        self.channels = channels
        self.L = L
        self.n_streams = n_streams      
        
        self.in_channels = config['in_channels']
        assert self.in_channels==len(self.channels), "Number of input channels is mismatched."
        
        self.chns_1 = config['chns_1']
        self.kernels_1 = config['kernels_1']
        self.act_1 = config['act_1']
        self.chns_1_aud = config['chns_1_aud']
        self.kernels_1_aud = config['kernels_1_aud']
        
        self.pool_stride_1 = config['pool_stride_1']       
        self.hidden_size = config['hidden_size']
        self.dropout = config['dropout']
        if "feature_freeze" in config.keys():
            self.feature_freeze = config['feature_freeze']
        else:
            self.feature_freeze = False        

        self.batchnorm_1 = nn.BatchNorm1d(self.in_channels+self.n_streams)
        self.inception_1_eeg = Inception1d(self.in_channels, self.chns_1, self.kernels_1, self.act_1)
        self.inception_1_aud = Inception1d(1, self.chns_1_aud, self.kernels_1_aud, self.act_1)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=self.pool_stride_1, padding=1)        
        x = torch.rand(1,self.in_channels, self.L)
        e = torch.rand(1,1, self.L)
        x = self.inception_1_eeg(x)
        e = self.inception_1_aud(e)
        
        (batch_size, self.combined_chns_eeg, _) = x.shape      
        (batch_size, self.combined_chns_aud, _) = e.shape
        self.combined_chns = self.combined_chns_eeg + self.n_streams*self.combined_chns_aud
        self.batchnorm_2 = nn.BatchNorm1d(self.combined_chns)
        self.dropout_1 = nn.Dropout(p=self.dropout)
        # classification layers
        if self.hidden_size > 0:
            self.fc1 = nn.Linear(self.n_streams*self.combined_chns_aud*self.combined_chns_eeg, self.hidden_size, bias=True)
            self.fc2 = nn.Linear(self.hidden_size, self.n_streams, bias=True)
        else:
            self.fc1 = nn.Linear(self.n_streams*self.combined_chns_aud*self.combined_chns_eeg, self.n_streams, bias=True)            
        
    def forward(self, eeg, env, y=None):
        (batch_size,c,_) = eeg.shape
        (_,n_streams,L) = env.shape
        assert self.n_streams==n_streams, "Number of streams of the input data is mismatched."
        # assert self.L==L, f"Number of samples of the input data is mismatched.{self.L} vs {L}"
        x = torch.cat((eeg[...,:L],env), dim=1) # batch_size,c+2,L   
        
        x = self.batchnorm_1(x)
        eeg = self.inception_1_eeg(x[:,:c,:])
        aud = x[:,-self.n_streams:,:]
        x = eeg
        for i in range(self.n_streams):
            envelope_i = self.inception_1_aud(aud[...,i:i+1,:])
            x = torch.cat((x,envelope_i), dim=1) # batch_size,c+n_streams,L
        # x = self.maxpool_1(x)
        x = self.batchnorm_2(x)
        envelope = x[:,-self.n_streams*self.combined_chns_aud:,:]
        eeg = x[:,:self.combined_chns_eeg,:]
        # correlation
        x = 1-CorrelationLoss()(envelope, eeg)
        x = self.dropout_1(x.view(batch_size, -1))
        if self.hidden_size > 0:
            x = F.elu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.fc1(x)
        
        return x    

    def initialize(self, state=None):
        for name, param in self.named_parameters():
            if param.requires_grad and 'weight' in name:
                if 'batchnorm' in name:
                    continue
                else:
                    nn.init.xavier_uniform_(param)
        if state is not None:
            self.load_state_dict(state)
        if self.feature_freeze:
            for param in self.batchnorm_1.parameters():
                param.requires_grad = False
            for param in self.inception_1_eeg.parameters():
                param.requires_grad = False
            for param in self.inception_1_aud.parameters():
                param.requires_grad = False                
            for param in self.batchnorm_2.parameters():
                param.requires_grad = False