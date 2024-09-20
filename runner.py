import os
import sys
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, GroupKFold
import pickle
import numpy as np
import torch
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from datetime import datetime
import multiprocessing as mp

from aadnet.loss import *
from aadnet.dataset import *
from utils.utils import *

def fit(model, criterion, optimizer, lr_scheduler, train_loader, valid_loader, epochs, devices, model_path, early_stop = 'loss', jobname = None, print_every=10):
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    device = devices[0] if (type(devices) is list) else devices
    model.to(device)
    best_loss_train = -1.0
    best_accr_train = 0.0
 
    best_loss_valid, best_accr_valid = evaluate(model, valid_loader, device, criterion, None, model_path=None, jobname=jobname, print_output=False)
    print(f'device {device} {datetime.now().time().replace(microsecond=0)} --- '
          f'Epoch: -1\t'
          f'Valid loss: {best_loss_valid:.8f}\t'
          f'Valid accuracy: {100 * best_accr_valid:.2f}')
    torch.save(model.state_dict(), model_path)
    print(f'device {device} Checkpoint saved at epoch -1.')
    early_stopping_waiting = 0
    patience = 5    
    for epoch in range(epochs):
        early_stopping_waiting += 1
        model.train()
        all_y_hat = []
        all_y_true = []        
        acc_reconst = []
        total_loss = 0.0
        for (eeg, aud, y_true) in train_loader:
            optimizer.zero_grad() # reset gradients
            (batch_size,n_streams,*_,L) = aud.shape
            if batch_size==1:
                continue
            eeg = eeg.to(device, dtype=torch.float)
            aud = aud.to(device, dtype=torch.float)
            y_true = y_true.to(device, dtype=torch.long)   
            y_hat = model(eeg, aud, y_true)
            if isinstance(criterion, CorrelationLoss): # for NSR model
                (batch_size,L) = y_hat.shape
                if L==1:
                    aud = aud[...,0].permute(1,0).unsqueeze(0)
                    y_hat = y_hat.permute(1,0).unsqueeze(0)
                    attd_aud = aud[:,y_true, np.arange(batch_size)].unsqueeze(1)
                    loss = criterion(y_hat, attd_aud)
                    acc_reconst.append(1-loss.data) # correlation of predicted and actual envelopes
                    loss = torch.mean(loss)
                else:
                    attd_aud = aud[np.arange(batch_size),y_true, :].unsqueeze(1)
                    y_hat = y_hat.unsqueeze(1)
                    loss = criterion(y_hat, attd_aud[...,:L])
                    loss = torch.mean(loss)
                    all_y_hat.append(1-criterion(y_hat, aud[...,:L]).data) # correlation of predicted and actual envelopes 
            elif isinstance(criterion, BCEWithLogitsLoss) or isinstance(criterion, BCELoss):
                loss = criterion(y_hat, one_hot(y_true, num_classes=n_streams).float()) 
                all_y_hat.append(y_hat.data)            
            else:
                loss = criterion(y_hat, y_true) 
                all_y_hat.append(y_hat.data)
            all_y_true.append(y_true.data)
            total_loss += loss.item()*batch_size
            loss.backward()
            optimizer.step()
            del eeg, aud, y_true, y_hat, loss
            torch.cuda.empty_cache()
        lr_scheduler.step()
        if isinstance(criterion, CorrelationLoss) and len(acc_reconst) > 1:
            acc = torch.mean(torch.hstack(acc_reconst)) 
        else:
            acc = accuracy(torch.cat(all_y_hat), torch.cat(all_y_true))
        epoch_loss = total_loss/len(train_loader.dataset)
        epoch_accr = acc.item()        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_accr)
        del all_y_hat, all_y_true, acc_reconst
        epoch_loss, epoch_accr = evaluate(model, valid_loader, device, criterion, None, None, jobname=jobname, print_output=False)
        valid_losses.append(epoch_loss)
        valid_accs.append(epoch_accr)
        
        if (early_stop=='loss') and ((valid_losses[-1] <= best_loss_valid) or epoch==0):
            best_loss_train = train_losses[-1]
            best_loss_valid = valid_losses[-1]
            best_accr_train = train_accs[-1]
            best_accr_valid = valid_accs[-1]            
            torch.save(model.state_dict(), model_path)
            early_stopping_waiting = 0
            print(f'device {device} Loss Checkpoint saved at epoch {epoch}.')
            
        if (early_stop=='accuracy') and ((valid_accs[-1] >= best_accr_valid) or epoch==0):
            best_loss_train = train_losses[-1]
            best_loss_valid = valid_losses[-1]
            best_accr_train = train_accs[-1]
            best_accr_valid = valid_accs[-1]            
            torch.save(model.state_dict(), model_path)
            early_stopping_waiting = 0
            print(f'device {device} Accuracy Checkpoint saved at epoch {epoch}.')
    
        if epoch % print_every == 0:
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_losses[epoch]:.8f}\t'
                  f'Valid loss: {valid_losses[epoch]:.8f}\t'
                  f'Train acc: {train_accs[epoch]:.8f}\t'
                  f'Valid acc: {valid_accs[epoch]:.8f}\t')

        torch.cuda.empty_cache()
        if (early_stopping_waiting > patience):
            break
    plt.clf()
    plt.plot(train_accs,'b-', label="train accuracy")
    plt.plot(valid_accs,'r.', label="validation accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    filepath = os.path.join(os.path.dirname(model_path), f"{jobname}_accuracy_curve.png") 
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    # plot loss    
    plt.clf()
    plt.plot(train_losses,'b-', label="train loss")
    plt.plot(valid_losses,'r.', label="validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('CEL loss')
    plt.legend()
    filepath = os.path.join(os.path.dirname(model_path), f"{jobname}_loss_curve.png") 
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate(model, data_loader, device, criterion, sr, model_path, jobname=None, print_output=False):
    '''
    Calculating accuracy of model
    '''
    if model_path is not None:
        print(f'loading: {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    if type(device) is list:
        device = device[0]
    model.device = device
    model.to(device)
    model.eval()
    total_loss = 0.0
    all_y_hat = []
    all_y_true = []
    acc_reconst = []
    for (eeg, aud, y_true) in data_loader:
        (batch_size,n_streams,*_,L) = aud.shape
        eeg = eeg.to(device, dtype=torch.float)
        aud = aud.to(device, dtype=torch.float)
        y_true = y_true.to(device, dtype=torch.long)
        with torch.no_grad():
            y_hat = model(eeg, aud, y_true)
            if isinstance(criterion, CorrelationLoss):
                (batch_size,L) = y_hat.shape
                if L==1:
                    aud = aud[...,0].permute(1,0).unsqueeze(0)
                    y_hat = y_hat.permute(1,0).unsqueeze(0)
                    attd_aud = aud[:,y_true, np.arange(batch_size)].unsqueeze(1)
                    loss = criterion(y_hat, attd_aud)
                    acc_reconst.append(1-loss.data) # correlation of predicted and actual envelopes
                    loss = torch.mean(loss)                    
                else:
                    attd_aud = aud[np.arange(batch_size),y_true, :].unsqueeze(1)
                    y_hat = y_hat.unsqueeze(1)
                    loss = criterion(y_hat, attd_aud[...,:L])
                    loss = torch.mean(loss)
                    all_y_hat.append(1-criterion(y_hat, aud[...,:L]).data) # correlation of predicted and actual envelopes                    
            elif isinstance(criterion, BCEWithLogitsLoss) or isinstance(criterion, BCELoss):
                loss = criterion(y_hat, one_hot(y_true, num_classes=n_streams).float())
                all_y_hat.append(y_hat.data)
            else:
                loss = criterion(y_hat, y_true) 
                all_y_hat.append(y_hat.data)   
            all_y_true.append(y_true.data)                     
        total_loss += loss.item()*batch_size
        torch.cuda.empty_cache()
        del eeg, aud, y_true, y_hat, loss
        
    # (TP,FP,TN,FN,acc,threshold) = metrics(torch.cat(all_y_hat), torch.cat(all_y_true), thresh=threshold, weighted=weighted)
    if isinstance(criterion, CorrelationLoss) and len(acc_reconst)>0:
        acc = torch.mean(torch.hstack(acc_reconst))
    else:
        acc = accuracy(torch.cat(all_y_hat), torch.cat(all_y_true))
    avg_loss = total_loss/len(data_loader.dataset)
    avg_accr = acc.item()
    
    return avg_loss,avg_accr
  
def trainLinearEnvelope(model_config, trainset, testset, windows, sr, sbj=None, fold=None, conn=None, step=0.0):  
    (eeg_tr, env_tr, evt_tr, groups_tr, y_tr) = trainset
    (eeg_te, env_te, evt_te, groups_te, y_te) = testset
    batch_tr = len(y_tr)
    attd_env_tr = env_tr[np.arange(batch_tr), y_tr]
    batch_te = len(y_te)
    attd_env_te = env_te[np.arange(batch_te), y_te]
    if model_config['model_name'] == 'LSQ':
        cv_gen = GroupKFold(n_splits=5).split(eeg_tr, attd_env_tr, groups=groups_tr)
        alpha_range = model_config['alpha_range']
        alpha_range = tuple(10**(i) for i in np.arange(alpha_range[0],alpha_range[1],alpha_range[2], dtype=float))
        model = RidgeCV(alphas=alpha_range, fit_intercept=True, scoring=pearson_scorer, cv=cv_gen, gcv_mode=None, store_cv_values=False)
        model.fit(eeg_tr, attd_env_tr)
    elif model_config['model_name'] == 'CCA':
        n_components = model_config['max_component']
        model_path = os.path.expandvars(model_config['model_path'])
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        file = os.path.abspath(os.path.join(model_path, f"{model_config['load_name']}_S_{sbj}_fold_{fold}.pkl"))
        if model_config['load_model']:            
            print(f'load_file: {file}')
            with open(file, 'rb') as f:
                model = pickle.load(f)
        else:
            model = CCA(n_components=n_components)
            print(f'Fitting CCA {n_components} components with eeg {eeg_tr.shape}, and envelope {attd_env_tr.shape}')
            (eeg_scores, aud_scores) = model.fit_transform(eeg_tr, attd_env_tr)
            if model_config['save_model']:
                print(f'save_file: {file}')
                with open(file,'wb') as f:
                    pickle.dump(model,f)                
    # evaluate
    train_accs = np.zeros(len(windows))
    test_accs = np.zeros(len(windows))
    step0 = 1 if int(step*sr)==0 else int(step*sr)
    for w in range(len(windows)):
        L = int(windows[w]*sr)
        step = max(step0, L//5)
        print(f'L = {L}')
        (eeg, env, y_true) = AADDataset.getWindowedConvAADData(eeg_te, env_te, y_te, L, step, groups_te)
        (te_samples,_,n_spks,*_) = env.shape
        score_tr = [[] for i in range(n_spks)]
        score_te = []        
        if model_config['model_name'] == 'LSQ':
            for i in range(n_spks):
                score_te.append(pearson_scorer(model, eeg, env[...,i,:]))
            score_te = np.array(score_te)
            y_hat = np.argmax(score_te, axis=0)
            test_accs[w] = (y_hat==y_true).astype(float).mean()
            del eeg, env, y_true
        elif model_config['model_name'] == 'CCA':
            for i in range(n_spks):
                (eeg_score, aud_score) = model.transform(eeg.reshape(te_samples*L,-1), env[...,i,:].reshape(te_samples*L,-1))
                eeg_score = eeg_score.reshape(te_samples,-1,n_components).transpose(0,2,1).reshape(te_samples*n_components,-1)
                aud_score = aud_score.reshape(te_samples,-1,n_components).transpose(0,2,1).reshape(te_samples*n_components,-1)
                r = np.diag(np.corrcoef(eeg_score, aud_score), te_samples*n_components).reshape(te_samples,n_components)
                score_te.append(r)
                del eeg_score, aud_score
            del eeg, env
            unique_trials, _ = np.unique(groups_tr, return_counts=True)
            tr_y_true = []
            for tr in unique_trials:
                (tr_eeg, tr_env, tr_y) = AADDataset.getWindowedConvAADData(eeg_tr[groups_tr==tr], env_tr[groups_tr==tr], y_tr[groups_tr==tr], L, step, groups_tr[groups_tr==tr])
                (n_samples,*_) = tr_env.shape
                tr_y_true.append(tr_y)
                for i in range(n_spks):
                    (eeg_score, aud_score) = model.transform(tr_eeg.reshape(n_samples*L,-1), tr_env[...,i,:].reshape(n_samples*L,-1))
                    eeg_score = eeg_score.reshape(n_samples,-1,n_components).transpose(0,2,1).reshape(n_samples*n_components,-1)
                    aud_score = aud_score.reshape(n_samples,-1,n_components).transpose(0,2,1).reshape(n_samples*n_components,-1)
                    r = np.diag(np.corrcoef(eeg_score, aud_score), n_samples*n_components).reshape(n_samples,n_components)
                    score_tr[i].append(r)
                    del eeg_score, aud_score
                del tr_eeg, tr_env, tr_y
            score_tr = [np.concatenate(score, axis=0) for score in score_tr]
            tr_y_true = np.concatenate(tr_y_true)
            # train a LDA classifier
            clf = LinearDiscriminantAnalysis()
            features_tr = np.array(score_tr)
            labels_tr = np.random.randint(n_spks, size=len(tr_y_true)) # randomize label of training set of LDA
            for i in range(len(tr_y_true)):
                features_tr[labels_tr[i],i],features_tr[tr_y_true[i],i] = features_tr[tr_y_true[i],i],features_tr[labels_tr[i],i] # swap data of attended speaker.
            features_te = np.array(score_te)
            labels_te = y_true
            
            # with cross-valdidation
            if model_config['opt']:
                cv = KFold(n_splits=5)
                acc = np.zeros(n_components)
                for c in range(n_components):
                    yhat = cross_val_predict(clf, np.concatenate(features_tr[...,:(c+1)], axis=1), labels_tr, cv=cv)
                    acc[c] = (yhat==labels_tr).astype(float).mean()
                c_opt = np.argmax(acc) + 1
            else:
                c_opts = model_config['fixed_component']
                c_opt = c_opts[sbj] if isinstance(c_opts, list) else c_opts
                
            # train clf again using c_opt and verify on testset
            print(f'c_opt: {c_opt}')
            clf.fit(np.concatenate(features_tr[...,:c_opt], axis=1), labels_tr)
            yhat_tr = clf.predict(np.concatenate(features_tr[...,:c_opt], axis=1))
            train_accs[w] = (yhat_tr==labels_tr).astype(float).mean()
            yhat_te = clf.predict(np.concatenate(features_te[...,:c_opt], axis=1))
            test_accs[w] = (yhat_te==labels_te).astype(float).mean()
            del features_tr, features_te, score_tr, score_te, tr_y_true, y_true, labels_tr, labels_te, yhat_tr, yhat_te
    
    if conn is None:
        print(f'train_accs: {train_accs}')
        print(f'test_accs: {test_accs}')
        return (train_accs, test_accs)
    else:
        conn.send((fold, train_accs, test_accs))

def parallelizeLSQ(config, data, nFold, sbj, parallelization=False):
    #parallelization
    cpuCount = os.cpu_count()
    print("Number of CPUs in the system:", cpuCount)
    processes = []
    parent_conn, child_conn = mp.Pipe()
    config.get(("learning", "nFold"), fallback=8)
    
    sr = config.get(("dataset", "sr"), fallback=64)
    windows = config.get(("dataset", "windows"))
    dataset_name = config.get(("dataset", "name"))
    step = config.get(("dataset", "step"), fallback=0.1)
    train_accs = np.zeros((len(windows), nFold))
    test_accs = np.zeros((len(windows), nFold))
    for fold in range(nFold):
        print(f'{datetime.now().time().replace(microsecond=0)} --- '
                    f'********* Fold {fold} **********')
        (train_data, test_data) = data[fold]
        trainset = eval(dataset_name).getTrialConvData(config, train_data[0], train_data[1], train_data[2])
        testset = eval(dataset_name).getTrialConvData(config, test_data[0], test_data[1], test_data[2])
        if parallelization:
            p = mp.Process(target=trainLinearEnvelope, args=(config.get('model'), trainset, testset, windows, sr, sbj, False, fold, child_conn, step))
            processes.append(p)
            p.start()
        else:
            (train_accs[:,fold], test_accs[:,fold]) = trainLinearEnvelope(config.get('model'), trainset, testset, windows, sr, sbj=sbj, fold=fold, step=step)
            print(f'test_accs: {test_accs}')
        del train_data, testset
    if parallelization:
        for p in processes:
            p.join()        
        while parent_conn.poll():
            (fold, train_acc, test_acc) = parent_conn.recv() 
            print(f'from fold {fold} train_acc={train_acc}, test_acc = {test_acc}')
            train_accs[:,fold] = train_acc
            test_accs[:,fold] = test_acc
        
    return train_accs, test_accs
