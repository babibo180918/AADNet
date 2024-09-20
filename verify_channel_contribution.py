import yaml
import sys
import os
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GroupKFold, train_test_split, cross_val_predict, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from copy import deepcopy

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

def verify_channel_contribution(config, jobname):
    setup_params = config.get('setup')
    output_path = os.path.abspath(setup_params['output_path'])
    trainModel = setup_params['trainModel']
    output_path = os.path.join(output_path, f"{config.get(('model','model_name'))}_{config.get(('dataset','name'))}")
    os.makedirs(output_path, exist_ok=True)

    dataset_params = config.get('dataset')
    channels = dataset_params['channels']
    nChns = len(channels)
    sr = dataset_params['sr']
    windows = dataset_params['windows']
    all_sbjs = np.array(dataset_params['all_sbjs'])
    num_sbjs = len(all_sbjs)
    from_sbj = dataset_params['from_sbj']
    to_sbj = dataset_params['to_sbj']    
    T = dataset_params['training_window'] # seconds
    L = int(sr*T) # sample
    n_streams = dataset_params['n_streams']    
    LChO_groups = dataset_params['LChO_groups']
    nGroups = len(LChO_groups)
    LChO_groups_idxs = []
    for gr in LChO_groups:
        idxs = [channels.index(ch) for ch in gr]
        LChO_groups_idxs.append(idxs)

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

    train_accs = np.zeros((len(windows), num_sbjs, nChns, nFold))
    test_accs = np.zeros((len(windows), num_sbjs, nChns, nFold))
    train_F1 = np.zeros((len(windows), num_sbjs, nChns, nFold))
    test_F1 = np.zeros((len(windows), num_sbjs, nChns, nFold))
        
    (split_tr, split_va, split_te) = tuple(dataset_params['splits'])
    
    for s in range(from_sbj, to_sbj):
        for i in range(nGroups):
            print(f'Leave channel {LChO_groups[i]} out.')
            channels_new = [ch for ch in channels if ch not in LChO_groups[i]]
            idxs = sorted(LChO_groups_idxs[i])
            config['dataset']['channels'] = channels_new
            crossSSData = eval(config.get(('dataset', 'name'))).createSSCrossValidation(subject=s, config=config)
            for fold in range(nFold):
                print(f'{datetime.now().time().replace(microsecond=0)} --- '
                    f'********** cross-training Sbj {s} - Fold {fold} **********')                                  
                # model
                lossClass = loss_params['name']
                criterion = eval(lossClass)()
                model = eval(model_params['model_name'])(model_params, L, n_streams, sr, channels)
                # dataset
                ((eegs_tr, envs_tr, y_tr),(eegs_te, envs_te, y_te)) = deepcopy(crossSSData[fold])
                (eegs_tr, eegs_va, envs_tr, envs_va, y_tr, y_va) = train_test_split(eegs_tr, envs_tr, y_tr, test_size=split_va, random_state=s)
                for j in idxs:
                    eegs_tr = [np.insert(trial, j, np.zeros((1,trial.shape[-1])), axis=0) for trial in eegs_tr]
                    eegs_va = [np.insert(trial, j, np.zeros((1,trial.shape[-1])), axis=0) for trial in eegs_va]
                    eegs_te = [np.insert(trial, j, np.zeros((1,trial.shape[-1])), axis=0) for trial in eegs_te]
                trainset = eval(config.get(('dataset', 'name')))(config, eegs_tr, envs_tr, y_tr)
                validset = eval(config.get(('dataset', 'name')))(config, eegs_va, envs_va, y_va)
                # dataloader
                trainLoader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                validLoader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                model_path = os.path.join(output_path, f"{model_params['model_name']}_LOCO_T_{T}_s_{s}_fold_{fold}.pth")
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
                for t in range(len(windows)):
                    validset.setWindowSize(windows[t])
                    validLoader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)       
                    loss, train_accs[t, s, i, fold]  = evaluate(model, validLoader, device, criterion, sr, model_path=model_path, jobname=f'{jobname}_SI_{s}_fold_{fold}_train', print_output=False)
                    testset.setWindowSize(windows[t])
                    testLoader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                    loss, test_accs[t, s, i, fold]  = evaluate(model, testLoader, device, criterion, sr, model_path=model_path, jobname=f'{jobname}_SI_{s}_fold_{fold}_test', print_output=False)
                    del validLoader, testLoader
                del trainset, validset, testset
            print(f'sbj {s} leave channel {LChO_groups[i]} test_accs: {np.mean(test_accs, -1, keepdims=False)[...,s,i]}')
            del crossSSData
        print(f'sbj {s} valid_accs: {np.mean(train_accs, -1, keepdims=False)[...,s,:]}')
        print(f'sbj {s} test_accs: {np.mean(test_accs, -1, keepdims=False)[...,s,:]}')
        
    train_accs = np.mean(train_accs, -1, keepdims=False)[...,from_sbj:to_sbj,:]
    test_accs = np.mean(test_accs, -1, keepdims=False)[...,from_sbj:to_sbj,:]    
    return train_accs, test_accs, train_F1, test_F1
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Verify leave-one-channel-out performance of pretrained models')
    parser.add_argument("-j", "--jobname", type=str, required=True, help="Name of running entity.")
    parser.add_argument("-c", "--configs", type=str, required=True, nargs='+', help="Config file path.")
    parser.add_argument("-v", "--verbose", type=bool, default=False, help="Enable DEBUG verbose mode.")
    args = parser.parse_args()
    jobname = args.jobname
    configs = []
    for p in args.configs:
        config = Config.load_config(os.path.abspath(p))
        configs.append(config)
    output_path = os.path.abspath(configs[0].get(('setup', 'output_path')))
    save_path = os.path.join(output_path, f"{jobname}_LOCO_acc.npy")    
    from_sbj = configs[0].get(('dataset', 'from_sbj'))
    to_sbj = configs[0].get(('dataset', 'to_sbj'))
    num_window = len(configs[0].get(('dataset', 'windows')))
    num_sbjs = len(configs[0].get(('dataset', 'all_sbjs')))
    num_chns = len(configs[0].get(('dataset', 'channels')))    
    
    LOCO_SI_test_accs = []
    for i in range(len(configs)):
        print(configs[i])
        (_, te_acc, _, _) = verify_channel_contribution(configs[i], jobname)
        LOCO_SI_test_accs.append(te_acc)
    LOCO_SI_test_accs = np.array(LOCO_SI_test_accs)
    #
    if os.path.isfile(save_path):
        all_LOCO_SI_test_accs = np.load(save_path)
    else:
        all_LOCO_SI_test_accs = np.zeros((len(configs), num_window, num_sbjs, num_chns))
    all_LOCO_SI_test_accs[:,:,from_sbj:to_sbj,:] = LOCO_SI_test_accs
    np.save(save_path, all_LOCO_SI_test_accs)
    print(f'all_LOCO_SI_test_accs: {all_LOCO_SI_test_accs}')
    
    
