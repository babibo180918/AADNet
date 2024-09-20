import os
import numpy as np
import torch
import torch.nn as nn


class DualTask_AE_Loss(nn.Module):
    def __init__(self, alpha1:float=0.3, alpha2:int=0.7, loss1=nn.MSELoss(), loss2=nn.BCELoss()):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.loss1 = loss1
        self.loss2 = loss2
    def forward(self, X_pred:torch.Tensor, y_hat:torch.Tensor, X:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
        return self.alpha1*self.loss1(X_pred, X) + self.alpha2*self.loss2(y_hat, y_true)
        
class WeightedFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=0.0):
        super().__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        targets = targets.type(torch.long)
        self.alpha = self.alpha.to(inputs.device)
        at = self.alpha.gather(0, targets.data.view(-1)).to(inputs.device)
        pt = torch.exp(-BCE_loss).to(inputs.device)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()   

class DualBCELoss(nn.Module):
    def __init__(self, loss_fn=nn.BCELoss()):
        super().__init__()
        self.loss_fn = loss_fn
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        targets = nn.functional.one_hot(targets, num_classes = inputs.shape[-1]).type(torch.FloatTensor).to(inputs.device)
        inputs = inputs.reshape(-1,1)
        targets = targets.reshape(-1,1)
        indices = torch.randperm(inputs.size()[0])
        loss = self.loss_fn(inputs[indices], targets[indices])
        return loss
        
class CorrelationLoss(nn.Module):
    def __init__(self, L=None, eps=1e-8):
        super().__init__()
        self.L = L
        self.eps = eps
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        (batch_size, *_, L) = targets.shape
        if self.L is not None and L > self.L:
            L = self.L
            
        in_mean = torch.mean(inputs, dim=-1, keepdim=True)
        tg_mean = torch.mean(targets, dim=-1, keepdim=True)
        num = 1. / inputs.shape[-1] * torch.bmm((inputs-in_mean), (targets-tg_mean).transpose(-1,-2))
        denom = torch.bmm(torch.std(inputs, dim=-1, keepdim=True), torch.std(targets, dim=-1, keepdim=True).transpose(-1,-2)) + self.eps
        return (1-num / denom).squeeze(dim=1)
        
def correlation(x, y, eps=1e-8):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    return 1-corr        