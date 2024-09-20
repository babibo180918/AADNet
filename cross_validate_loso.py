import yaml
import sys
import os
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GroupKFold, train_test_split, cross_val_predict, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification

import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import *
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import StepLR
from torch.profiler import profile, record_function, ProfilerActivity
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from runner import *
from aadnet.EnvelopeAAD import *
from aadnet.dataset import *
from aadnet.loss import *
from utils.utils import *
from utils.config import Config

def trainLOSO(config, jobname):
    setup_params = config.get('setup')
    output_path = os.path.abspath(setup_params['output_path'])
    trainModel = setup_params['trainModel']
    output_path = os.path.join(output_path, f"{config.get(('model','model_name'))}_{config.get(('dataset','name'))}")
    os.makedirs(output_path, exist_ok=True)

    dataset_params = config.get('dataset')
    channels = dataset_params['channels']
    sr = dataset_params['sr']
    windows = dataset_params['windows']
    all_sbjs = np.array(dataset_params['all_sbjs'])
    num_sbjs = len(all_sbjs)
    from_sbj = dataset_params['from_sbj']
    to_sbj = dataset_params['to_sbj']    
    T = dataset_params['training_window'] # seconds
    L = int(sr*T) # sample
    n_streams = dataset_params['n_streams']    

    model_params = config.get('model')
    
    learning_params = config.get('learning')
    optimizer_params = learning_params['optimizer']        
    loss_params = learning_params['loss_function'] 
    optimizer_params = learning_params['optimizer']
    running_params = learning_params['running']
    loss_params = learning_params['loss_function']
    threshold = learning_params['threshold']
    nFold = learning_params['nFold']
    
    batch_size = running_params['batch_size']
    num_workers = running_params['num_workers']
    epochs = running_params['epochs']
    
    print_every = running_params['print_every']
    devices = running_params['device'] 
    device = devices[0] if (type(devices) is list) else devices    
    if not torch.cuda.is_available():
        device = 'cpu'
    
    early_stop = running_params['early_stop']
    opt = optimizer_params['opt']
    lr = float(optimizer_params['lr'])
    lr_decay_step = optimizer_params['lr_decay_step']
    lr_decay_gamma = optimizer_params['lr_decay_gamma']
    weight_decay = float(optimizer_params['weight_decay'])

    train_accs = np.zeros((len(windows), num_sbjs, nFold))
    test_accs = np.zeros((len(windows), num_sbjs, nFold))
    train_F1 = np.zeros((len(windows), num_sbjs, nFold))
    test_F1 = np.zeros((len(windows), num_sbjs, nFold))
        
    (split_tr, split_va, split_te) = tuple(dataset_params['splits'])
    for s in range(from_sbj, to_sbj):
        crossSIData = eval(config.get(('dataset', 'name'))).createSICrossValidation(subject=s, config=config)
        if model_params['model_name'] in ['LSQ', 'CCA']:
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                    f'********** cross-training Sbj {s} **********')
            train_accs[:,s,:], test_accs[:,s,:] = parallelizeLSQ(config, crossSIData, nFold, s, parallelization=running_params['parallelization'])
        else:
            for fold in range(nFold):
                print(f'{datetime.now().time().replace(microsecond=0)} --- '
                    f'********** cross-training Sbj {s} - Fold {fold} **********')                                  
                # model
                lossClass = loss_params['name']
                criterion = eval(lossClass)()
                model = eval(model_params['model_name'])(model_params, L, n_streams, sr, channels)
                # dataset
                ((eegs_tr, envs_tr, y_tr),(eegs_te, envs_te, y_te)) = crossSIData[fold]
                (eegs_tr, eegs_va, envs_tr, envs_va, y_tr, y_va) = train_test_split(eegs_tr, envs_tr, y_tr, test_size=split_va, random_state=s)                
                trainset = eval(config.get(('dataset', 'name')))(config, eegs_tr, envs_tr, y_tr)
                validset = eval(config.get(('dataset', 'name')))(config, eegs_va, envs_va, y_va)
                print(f'trainset: {len(trainset)}')
                print(f'validset: {len(validset)}')
                # dataloader
                trainLoader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                validLoader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                model_path = os.path.join(output_path, f"{model_params['model_name']}_SI_T_{T}_s_{s}_fold_{fold}.pth")
                if model_params['pretrained'] is not None:
                    pretrained_path = os.path.abspath(f"{model_params['pretrained']}_T_{T}_s_{s}_fold_{fold}.pth")
                    pretrained = torch.load(pretrained_path, map_location=torch.device(device))
                else:
                    pretrained = None
                model.initialize(pretrained)
                if trainModel:
                    optimizer = eval(opt)(model.parameters(), lr=lr, weight_decay=weight_decay)
                    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
                    fit(model, criterion, optimizer, scheduler, trainLoader, validLoader, epochs, device, model_path=model_path, early_stop=early_stop, jobname=f'{jobname}_SI_{s}_fold_{fold}', print_every=1)
                    del optimizer, scheduler
                else:
                    model_path = None
                # evaluate
                testset = eval(config.get(('dataset', 'name')))(config, eegs_te, envs_te, y_te)
                print(f'testset: {len(testset)}')
                for t in range(len(windows)):
                    validset.setWindowSize(windows[t])
                    validLoader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)       
                    loss, train_accs[t, s, fold]  = evaluate(model, validLoader, device, criterion, sr, model_path=model_path, jobname=f'{jobname}_SI_{s}_fold_{fold}_train', print_output=False)
                    testset.setWindowSize(windows[t])
                    testLoader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                    loss, test_accs[t, s, fold]  = evaluate(model, testLoader, device, criterion, sr, model_path=model_path, jobname=f'{jobname}_SI_{s}_fold_{fold}_test', print_output=False)
                    del validLoader, testLoader
                print(f'sbj {s} fold_valid: loss={loss}, acc={train_accs[:, s, fold]}')
                print(f'sbj {s} fold_test: loss={loss}, acc={test_accs[:, s, fold]}')
                del trainset, validset, testset, eegs_tr, eegs_va, eegs_te, envs_tr, envs_va, envs_te
        del crossSIData
        print(f'sbj {s} valid_accs: {np.mean(train_accs, -1, keepdims=False)[...,s]}')
        print(f'sbj {s} test_accs: {np.mean(test_accs, -1, keepdims=False)[...,s]}')
        
    train_accs = np.mean(train_accs, -1, keepdims=False)[...,from_sbj:to_sbj]
    test_accs = np.mean(test_accs, -1, keepdims=False)[...,from_sbj:to_sbj]    
    print(f'valid_accs: {train_accs}')
    print(f'test_accs: {test_accs}')    
    
    return train_accs, test_accs, train_F1, test_F1
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Leave-one-subject-out cross validation.')
    parser.add_argument("-j", "--jobname", type=str, required=True, help="Name of training entity.")
    parser.add_argument("-c", "--configs", type=str, required=True, nargs='+', help="Config file path.")
    parser.add_argument("-v", "--verbose", type=bool, default=False, help="Enable DEBUG verbose mode.")
    args = parser.parse_args()
    jobname = args.jobname
    configs = []
    for p in args.configs:
        config = Config.load_config(os.path.abspath(p))
        configs.append(config)
    output_path = os.path.abspath(configs[0].get(('setup', 'output_path')))
    save_path = os.path.join(output_path, f"{jobname}_SI_acc.npy")    
    from_sbj = configs[0].get(('dataset', 'from_sbj'))
    to_sbj = configs[0].get(('dataset', 'to_sbj'))
    
    SI_test_accs = []
    for i in range(len(configs)):
        print(configs[i])
        (_, te_acc, _, _) = trainLOSO(configs[i], jobname)
        SI_test_accs.append(te_acc)
    SI_test_accs = np.array(SI_test_accs)
    #
    if os.path.isfile(save_path):
        all_SI_test_accs = np.load(save_path)
    else:
        num_window = len(configs[0].get(('dataset', 'windows')))
        num_sbjs = len(configs[0].get(('dataset', 'all_sbjs')))
        all_SI_test_accs = np.zeros((len(configs), num_window, num_sbjs))
    all_SI_test_accs[:,:,from_sbj:to_sbj] = SI_test_accs
    np.save(save_path, all_SI_test_accs)
    print(f'all_SI_test_accs: {all_SI_test_accs}')
    
    
