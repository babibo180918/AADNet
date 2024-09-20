from abc import ABC, abstractmethod
import torch.nn as nn

class BaseNet(ABC, nn.Module):
    
    @abstractmethod
    def __init__(self):
        nn.Module.__init__(self)
        pass
    
    @abstractmethod
    def forward(self, x):
        pass
        
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def fit(self):
        pass
       
    @abstractmethod
    def evaluate(self):
        pass
    
