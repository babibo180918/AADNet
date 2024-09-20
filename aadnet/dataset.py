import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import json
import joblib
import scipy.io as spio
from scipy.io import loadmat
from scipy.io.wavfile import read
from sklearn.model_selection import KFold, GroupKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import mne
from mne_bids import read_raw_bids, BIDSPath
from mne.channels import read_custom_montage

from .loss import *
from utils.config import Config
from preprocessing import preprocessing, preprocessing_linear, preprocessing_cnn

def make_conv(X, env, evt=None,  eeg_context=1, aud_context=1, padding=0):
    assert X.shape[-1]==env.shape[-1], "Size of inputs are mismatched."    
    (num_ch,L) = X.shape
    (au_ch,_) = env.shape
    # Select non-nan data and desired eeg ch
    X = X.astype(np.float32)
    env = env.astype(np.float32)
    #
    idx_keep = np.ones(L, dtype=bool)
    for i in range(num_ch):
        idx_keep = idx_keep&(~np.isnan(X[i,:]))
    X = X[:, idx_keep]
    env = env[:, idx_keep]
    #
    X = np.pad(X, pad_width=((0,0),(0,eeg_context-1)), constant_values=padding)
    env = np.pad(env, pad_width=((0,0),(aud_context-1,0)), constant_values=padding)

    # Create output:  
    num_output = X.shape[-1] - eeg_context + 1
    X_out = np.nan * np.ones((num_output, eeg_context*num_ch))
    env_out = np.nan * np.ones((num_output, au_ch, aud_context))
    for idx in range(num_output):
        eeg_idx_keep = idx + np.arange(eeg_context)
        X_out[idx] = np.ravel(X[:, eeg_idx_keep])
        aud_idx_keep = idx + np.arange(aud_context)
        env_out[idx] = env[:, aud_idx_keep]
    
    return X_out, env_out

class AADDataset(Dataset):
    all_data = None
    
    def __init__(self, config, eegs, audios, labels, **kwargs):
        assert len(eegs)==len(audios) and len(eegs) == len(labels), "Number of trials are mismatched in the input data."
        self.config = config
        self.sr = self.config.get(('dataset', 'sr'))
        self.T = self.config.get(('dataset', 'training_window')) # seconds
        self.L = int(self.T*self.sr) # samples
        self.step = int(self.config.get(('dataset', 'step'))*self.sr) # samples
        if self.step==0:
            self.step = 1
        self.duplicate = self.config.get(('dataset', 'duplicate'), fallback=True) # seconds
        self.eegs = eegs
        self.audios = audios
        self.labels = labels
        self.n_trials = len(self.eegs)
        self.data_size = 0
        self.__count_data_points_()        
        
    def __count_data_points_(self):
        """
        Count number of possible epoch data in each trial for a fast access.
        """
        self.data_points = []
        self.trial_lut = []        
        for i in range(self.n_trials):
            trial_len = self.eegs[i].shape[-1]
            n_points = int((trial_len-self.L)/self.step) + 1
            self.data_points.append(n_points)
            self.trial_lut = self.trial_lut + [i for n in range(n_points)]
        self.data_size = len(self.trial_lut)
    
    def __len__(self):
        return (2*self.data_size if self.duplicate else self.data_size)
        
    def __getitem__(self, idx):
        if idx>=self.data_size:
            idx = idx - self.data_size
            (trial, loc) = self.__parse_index__(idx)
            (eeg, audio, y) = self.__get_item_at__(trial, loc)
            audio = audio[[1,0]]
            y = 1-y 
        else:
            (trial, loc) = self.__parse_index__(idx)
            (eeg, audio, y) = self.__get_item_at__(trial, loc)
        return torch.tensor(eeg), torch.tensor(audio), torch.tensor(y)
        
    def __get_item_at__(self, trial, loc):    
        start = loc*self.step
        eeg = self.eegs[trial][...,start:start+self.L]
        audio = self.audios[trial][...,start:start+self.L]
        y = self.labels[trial]
        return eeg, audio, y
        
    def __parse_index__(self, idx):
        trial_idx = self.trial_lut[idx]
        loc_idx = idx - sum(self.data_points[:trial_idx])
        return trial_idx,loc_idx  
    
    def setWindowSize(self, size):
        self.T = size # seconds
        self.L = int(self.T*self.sr) # samples
        self.step = int(self.L/2) # samples
        self.__count_data_points_()

    @classmethod
    def loadData(__cls__, config: Config) -> None:
        """Load all data from the raw dataset"""
        raise NotImplementedError

    @classmethod
    def createSSCrossValidation(__cls__, config: Config) -> tuple[np.ndarray, np.ndarray, ...]:
        """Create train and test data for subject-specific validation"""
        raise NotImplementedError            

    @classmethod
    def createSICrossValidation(__cls__, config: Config) -> tuple[np.ndarray, np.ndarray, ...]:
        """Create train and test data for subject-independent validation"""
        raise NotImplementedError

    @classmethod
    def getPooledData(__cls__, config: Config) -> list[tuple]:
        """Create train and test data for subject-independent validation"""
        all_sbjs = np.array(config.get(("dataset", "all_sbjs")))
        __cls__.loadData(config)
        pooledData = []        
        for s in all_sbjs:
            s_eeg, s_aud, s_label = [], [], []
            s_data = __cls__.all_data[s]
            for trial in range(len(s_data)):
                label = s_data[trial]['label']
                s_eeg.append(s_data[trial]['eeg'])
                s_aud.append(s_data[trial]['audio'])
                s_label.append(label)
            pooledData.append((s_eeg, s_aud, s_label))        
        return pooledData
        
    @classmethod
    def preprocess(__cls__, config: Config) -> tuple[np.ndarray, np.ndarray, ...]:
        """preprocess the data"""
        raise NotImplementedError        
        
    @classmethod    
    def getTrialConvData(__cls__, config, eegs, envs, y_trues) -> tuple[np.ndarray, np.ndarray, ...]:
        chns = __cls__.channels        
        eeg_context = round(config.get(('dataset', 'eeg_context'), fallback=0.4)*config.get(('dataset', 'sr'), fallback=64)) + 1
        aud_context = round(config.get(('dataset', 'aud_context'), fallback=0)*config.get(('dataset', 'sr'), fallback=64)) + 1
        ds_chns = np.array(config.get(('dataset', 'channels'), fallback=None))
        selected_chns = [list(chns).index(ch) for ch in ds_chns]
        assert len(eegs)==len(envs) and len(eegs) == len(y_trues), "Number of trials are mismatched in the input data."
        n_trials = len(y_trues)
        groups_all = []
        eeg_all = []
        env_all = []
        y_all = []  
        for i in range(n_trials):
            eeg = eegs[i][selected_chns]
            env = envs[i]
            eeg, env = make_conv(eeg, env, eeg_context=eeg_context, aud_context=aud_context)
            groups = i * np.ones(eeg.shape[0])
            eeg_all.append(eeg)
            env_all.append(env)
            y_all.append(np.full(len(env), y_trues[i], dtype=np.int32))
            groups_all.append(groups)
            del eeg, env
                
        eeg_all = np.concatenate(eeg_all, axis=0)
        env_all = np.concatenate(env_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        groups_all = np.concatenate(groups_all, axis=0)
        eeg_shape = eeg_all.shape
        aud_shape = env_all.shape
        
        scaler_path = config.get(('dataset', 'scaler', 'path'), fallback=None)
        scaler = None
        if scaler_path is not None:
            scaler_path = os.path.expandvars(scaler_path)
            if os.path.exists(scaler_path):
                print(f'Loading scaler: {scaler_path}')
                scaler = joblib.load(scaler_path)
            else:    
                if config.get(('dataset', 'scaler', 'type')) == 'MinMaxScaler':
                    feature_range = tuple(config.get(('dataset', 'scaler', 'feature_range')))
                    scaler = MinMaxScaler(feature_range=feature_range)
                elif config.get(('dataset', 'scaler', 'type')) == 'RobustScaler':
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))   
                all_data = []
                for s_data in __cls__.all_data:
                    for trial_data in s_data:
                        all_data.append(trial_data['eeg'].flatten())
                all_data = np.concatenate(all_data)
                scaler.fit_transform(all_data.reshape(-1,1))
                del all_data
                joblib.dump(scaler, scaler_path)
        if scaler is not None:
            eeg_all = scaler.transform(eeg_all.reshape(-1,1)).reshape(eeg_shape)
                           
        return eeg_all, env_all, env_all, groups_all, y_all
    
    @classmethod
    def getWindowedConvAADData(__cls__, eegs, envs, y_trues, L, step, trial_groups=None):
        unique_trials, counts = np.unique(trial_groups, return_counts=True)
        w_eegs = []
        w_envs = []
        w_y = []
        for i in unique_trials:
            eeg = eegs[trial_groups==i]
            env = envs[trial_groups==i]
            y = y_trues[trial_groups==i]
            trial_len = eeg.shape[0]
            n_chunks = int((trial_len-L)/step) + 1
            start = 0
            end = start + L
            for j in range(n_chunks):
                w_eegs.append(eeg[j*step:j*step+L])
                w_envs.append(env[j*step:j*step+L])
                w_y.append(y[0])
            del eeg, env, y
        return np.array(w_eegs), np.array(w_envs), np.array(w_y)

class EventAADDataset(AADDataset):
    TRIAL_START_CODE = 8
    TRIAL_END_CODE = 9
    ordinary_channels = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7-T3', 'C3', 'Cz', 'C4', 'T8-T4', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7-T5', 'P3', 'Pz', 'P4', 'P8-T6', 'POz', 'O1', 'Oz', 'O2']
    def __init__(self, config, eegs, envs, labels, **kwargs):
        super().__init__(config, eegs, envs, labels, **kwargs)        
        
    @classmethod
    def preprocess(__cls__, config: Config) -> tuple[np.ndarray, np.ndarray, ...]:
        """preprocess the data"""
        raise NotImplementedError   

    @classmethod
    def loadData(__cls__, config: Config, subject=None) -> None:
        """Load all data from the raw dataset and pre-process."""
        data_folder = os.path.expandvars(config.get(("dataset", "folder"), fallback=''))
        stimuli_path = os.path.expandvars(config.get(("dataset", "stimuli_path"), fallback=''))
        data_files = config.get(("dataset", "pre_processed"), fallback=None)
        preproc_pipeline = config.get(("dataset", "preprocess"), fallback="linear")
        target_sr = config.get(("dataset", "sr"), fallback=64)
        target_sr_aud = target_sr
        __cls__.selected_chs = np.array(config.get(("dataset", "channels")))
        if data_files is None:
            data_files = config.get(("dataset", "raw"), fallback=None)      
        all_sbjs = np.array(config.get(("dataset", "all_sbjs")))
        if subject is None:
            subjects = all_sbjs
        else:
            subjects = [subject]
        if __cls__.all_data is None:
            if preproc_pipeline == "ThorntonM":
                preproc = preprocessing_cnn.preprocess
            elif preproc_pipeline == "linear":
                preproc = preprocessing_linear.preprocess
            else:
                preproc = preprocessing.preprocess
            __cls__.all_data = []
            for sbj in subjects:
                f = data_files[sbj]
                print(f'Loading subject {f}')                    
                bids_path = BIDSPath(subject=f, task='speech~speech~attend', root=data_folder)
                raw = read_raw_bids(bids_path=bids_path, verbose='CRITICAL')
                raw.set_eeg_reference(ref_channels=__cls__.ordinary_channels)
                raw.set_eeg_reference(ref_channels=list(__cls__.selected_chs))
                __cls__.sr_eeg = raw.info['sfreq']
                __cls__.channels = raw.ch_names
                selected_idx = [__cls__.channels.index(ch) for ch in __cls__.selected_chs]                    
                events_path = (str(bids_path.fpath)).replace('eeg.set', 'events.tsv')
                events_df = pd.read_csv(events_path, sep='\t', engine='python')[['onset', 'value', 'attended_stimulus']]
                events_df['onset'] = (events_df['onset']*__cls__.sr_eeg).astype(int)
                events = events_df[(events_df['value']==__cls__.TRIAL_START_CODE)|(events_df['value']==__cls__.TRIAL_END_CODE)].to_numpy()
                
                events_stimuli_path = events_path.replace('.tsv', '.json')
                with open(events_stimuli_path, 'r') as jsonFile:
                    stimuli_info = json.load(jsonFile)
                s_data = []
                for trial in range(len(events)//2):
                    label = events[2*trial][2]
                    stimuli = [os.path.basename(sti.replace('\\', '/')) for sti in stimuli_info[trial]['stimulies']]
                    eeg = raw.get_data(picks=__cls__.selected_chs, start=events[2*trial][0], stop=events[2*trial+1][0])
                    audio = []
                    sr_aud = []
                    for s in stimuli:
                        sr, a = read(os.path.join(stimuli_path, s))
                        audio.append(a)
                        sr_aud.append(sr)
                    eeg,audio = preproc(eeg, audio, __cls__.sr_eeg, sr_aud, target_sr, target_sr_aud)
                    s_data.append({'eeg':eeg, 'audio':audio, 'label':label, 'stimuli':stimuli})
                    del eeg,audio
                del raw
                if subject is None:
                    __cls__.all_data.append(s_data)
                else:
                    __cls__.all_data = None 
                    return s_data

    @classmethod
    def createSSCrossValidation(__cls__, subject, config: Config) -> list[tuple]:
        """Create train and test data for subject-specific validation"""
        preproc_pipeline = config.get(("dataset", "preprocess"), fallback="linear")
        nFold = config.get(("learning", "nFold"), fallback=8)
        subject_data=__cls__.loadData(config, subject)
        #
        crossSSData = []
        n_trials = len(subject_data)
        kf = KFold(n_splits=nFold, random_state=subject, shuffle=True)
        trial_idx_splits = [(train,test) for train,test in kf.split(range(n_trials))]         
        for fold in range(nFold):
            train_idxs = trial_idx_splits[fold][0]
            test_idxs = trial_idx_splits[fold][1]
            test_attended_stimuli = []
            tr_eeg, tr_aud, tr_label = [], [], []
            te_eeg, te_aud, te_label = [], [], []
            for trial in test_idxs:
                label = subject_data[trial]['label']
                te_eeg.append(subject_data[trial]['eeg'])
                te_aud.append(subject_data[trial]['audio'])
                te_label.append(label)
            for trial in train_idxs:
                label = subject_data[trial]['label']
                tr_eeg.append(subject_data[trial]['eeg'])
                tr_aud.append(subject_data[trial]['audio'])
                tr_label.append(label)
            crossSSData.append(((tr_eeg, tr_aud, tr_label),(te_eeg, te_aud, te_label)))        
        
        return crossSSData
        
    @classmethod
    def createSICrossValidation(__cls__, subject, config: Config) -> list[tuple]:
        """Create train and test data for subject-independent validation"""
        preproc_pipeline = config.get(("dataset", "preprocess"), fallback="linear")
        all_sbjs = np.array(config.get(("dataset", "all_sbjs")))
        nFold = config.get(("learning", "nFold"), fallback=8)            
        num_sbjs = len(all_sbjs)
        __cls__.loadData(config)
        #
        crossSIData = []
        tr_sbjs = np.delete(all_sbjs, np.where(all_sbjs==subject)[0])
        test_data = __cls__.all_data[subject]
        n_trials = len(test_data)
        kf = KFold(n_splits=nFold, random_state=subject, shuffle=True)
        trial_idx_splits = [(train,test) for train,test in kf.split(range(n_trials))]         
        for fold in range(nFold):
            test_idxs = trial_idx_splits[fold][1]
            test_attended_stimuli = []
            tr_eeg, tr_aud, tr_label = [], [], []
            te_eeg, te_aud, te_label = [], [], []
            for trial in test_idxs:
                label = test_data[trial]['label']
                attd_stimuliname = test_data[trial]['stimuli'][label]
                test_attended_stimuli.append(attd_stimuliname)
                te_eeg.append(test_data[trial]['eeg'])
                te_aud.append(test_data[trial]['audio'])
                te_label.append(label)
            for tr_s in tr_sbjs:
                s_data = __cls__.all_data[tr_s]
                for trial in range(len(s_data)):
                    label = s_data[trial]['label']
                    attd_stimuliname = s_data[trial]['stimuli'][label]
                    if any(i in attd_stimuliname for i in test_attended_stimuli):
                        continue
                    else:
                        tr_eeg.append(s_data[trial]['eeg'])
                        tr_aud.append(s_data[trial]['audio'])
                        tr_label.append(label)
            crossSIData.append(((tr_eeg, tr_aud, tr_label),(te_eeg, te_aud, te_label)))        
        
        return crossSIData
        
class DTUDataset(AADDataset):
    ordinary_channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
    def __init__(self, config, eegs, envs, labels, **kwargs):
        super().__init__(config, eegs, envs, labels, **kwargs)        
        
    @classmethod
    def preprocess(__cls__, config: Config) -> tuple[np.ndarray, np.ndarray, ...]:
        """preprocess the data"""
        raise NotImplementedError        
        
    @classmethod
    def loadData(__cls__, config: Config, subject=None) -> None:
        """Load all data from the raw dataset and pre-process."""
        data_folder = os.path.expandvars(config.get(("dataset", "folder"), fallback=''))
        stimuli_path = os.path.expandvars(config.get(("dataset", "stimuli_path"), fallback=''))
        data_files = config.get(("dataset", "pre_processed"), fallback=None)
        preproc_pipeline = config.get(("dataset", "preprocess"), fallback="linear")
        target_sr = config.get(("dataset", "sr"), fallback=64)
        target_sr_aud = target_sr
        __cls__.selected_chs = np.array(config.get(("dataset", "channels")))
        
        if data_files is None:
            data_files = config.get(("dataset", "raw"), fallback=None)        
        all_sbjs = np.array(config.get(("dataset", "all_sbjs")))
        if subject is None:
            subjects = all_sbjs
        else:
            subjects = [subject]
        if __cls__.all_data is None:
            if preproc_pipeline == "ThorntonM":
                preproc = preprocessing_cnn.preprocess
            elif preproc_pipeline == "linear":
                preproc = preprocessing_linear.preprocess
            else:
                preproc = preprocessing.preprocess       
            __cls__.all_data = []
            data_files = [os.path.join(data_folder, f) for f in data_files]
            for sbj in subjects:
                f = data_files[sbj]
                print(f'Loading data file {f}')
                s_data = []
                mat_data = loadmat(f, squeeze_me=True)
                raw = mat_data['data']
                info = mat_data['expinfo']
                __cls__.sr_eeg = raw['fsample'].item()['eeg'].item()
                __cls__.channels = raw['dim'].item()['chan'].item()['eeg'].item().tolist()
                __cls__.ordinary_selected_idx = [__cls__.channels.index(ch) for ch in __cls__.ordinary_channels]
                selected_idx = [__cls__.channels.index(ch) for ch in __cls__.selected_chs]
                events = raw['event'].item()['eeg'].item()['sample'].item()
                events = np.append(events, 2*events[-1])
                for trial in range(len(events)//2):
                    if len(info['wavfile_female'][trial]) == 0:
                       continue
                    smpl = events[2*trial]
                    ref = raw['eeg'].item()[events[2*trial]:events[2*(trial+1)],__cls__.ordinary_selected_idx].T.mean(axis=0, keepdims=True)
                    eeg = raw['eeg'].item()[events[2*trial]:events[2*(trial+1)],selected_idx].T-ref
                    label = info['attend_mf'][trial]-1
                    stimuli = [info['wavfile_male'][trial], info['wavfile_female'][trial]]
                    audio = []
                    sr_aud = []
                    for s in stimuli:
                        sr, a = read(os.path.join(stimuli_path, s))
                        audio.append(a)
                        sr_aud.append(sr)
                    eeg,audio = preproc(eeg, audio, __cls__.sr_eeg, sr_aud, target_sr, target_sr_aud)
                    s_data.append({'eeg':eeg, 'audio':audio, 'label':label, 'stimuli':stimuli})
                    del eeg,audio
                del raw, info
                if subject is None:
                    __cls__.all_data.append(s_data)
                else:
                    __cls__.all_data = None 
                    return s_data
        
    @classmethod
    def createSSCrossValidation(__cls__, subject, config: Config) -> list[tuple]:
        """Create train and test data for subject-specific validation"""
        preproc_pipeline = config.get(("dataset", "preprocess"), fallback="linear")
        nFold = config.get(("learning", "nFold"), fallback=8)
        subject_data=__cls__.loadData(config, subject)
        #
        crossSSData = []
        n_trials = len(subject_data)
        kf = KFold(n_splits=nFold, random_state=subject, shuffle=True)
        trial_idx_splits = [(train,test) for train,test in kf.split(range(n_trials))]         
        for fold in range(nFold):
            train_idxs = trial_idx_splits[fold][0]
            test_idxs = trial_idx_splits[fold][1]
            tr_eeg, tr_aud, tr_label = [], [], []
            te_eeg, te_aud, te_label = [], [], []
            for trial in test_idxs:
                label = subject_data[trial]['label']
                te_eeg.append(subject_data[trial]['eeg'])
                te_aud.append(subject_data[trial]['audio'])
                te_label.append(label)
            for trial in train_idxs:
                label = subject_data[trial]['label']
                tr_eeg.append(subject_data[trial]['eeg'])
                tr_aud.append(subject_data[trial]['audio'])
                tr_label.append(label)
            crossSSData.append(((tr_eeg, tr_aud, tr_label),(te_eeg, te_aud, te_label)))        
        
        return crossSSData

    @classmethod
    def createSICrossValidation(__cls__, subject, config: Config) -> list[tuple]:
        """Create train and test data for subject-independent validation"""
        preproc_pipeline = config.get(("dataset", "preprocess"), fallback="linear")
        all_sbjs = np.array(config.get(("dataset", "all_sbjs")))
        nFold = config.get(("learning", "nFold"), fallback=8)            
        num_sbjs = len(all_sbjs)
        __cls__.loadData(config)
        #
        crossSIData = []
        tr_sbjs = np.delete(all_sbjs, np.where(all_sbjs==subject)[0])
        test_data = __cls__.all_data[subject]
        n_trials = len(test_data)
        kf = KFold(n_splits=nFold, random_state=subject, shuffle=True)
        trial_idx_splits = [(train,test) for train,test in kf.split(range(n_trials))]         
        for fold in range(nFold):
            test_idxs = trial_idx_splits[fold][1]
            test_attended_stimuli = []
            tr_eeg, tr_aud, tr_label = [], [], []
            te_eeg, te_aud, te_label = [], [], []
            for trial in test_idxs:
                label = test_data[trial]['label']
                attd_stimuliname = test_data[trial]['stimuli'][label]
                test_attended_stimuli.append(attd_stimuliname)
                te_eeg.append(test_data[trial]['eeg'])
                te_aud.append(test_data[trial]['audio'])
                te_label.append(label)
            for tr_s in tr_sbjs:
                s_data = __cls__.all_data[tr_s]
                for trial in range(len(s_data)):
                    label = s_data[trial]['label']
                    attd_stimuliname = s_data[trial]['stimuli'][label]
                    if any(i in attd_stimuliname for i in test_attended_stimuli):
                        continue
                    else:
                        tr_eeg.append(s_data[trial]['eeg'])
                        tr_aud.append(s_data[trial]['audio'])
                        tr_label.append(label)
            crossSIData.append(((tr_eeg, tr_aud, tr_label),(te_eeg, te_aud, te_label)))        
        
        return crossSIData

class Das2019Dataset(AADDataset):  
    ordinary_channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
    
    def __init__(self, config, eegs, envs, labels, **kwargs):
        super().__init__(config, eegs, envs, labels, **kwargs)        
        
    @classmethod
    def preprocess(__cls__, config: Config) -> tuple[np.ndarray, np.ndarray, ...]:
        """preprocess the data"""
        raise NotImplementedError  

    @classmethod
    def loadData(__cls__, config: Config, subject=None) -> None:
        """Load all data from the raw dataset and pre-process."""
        data_folder = os.path.expandvars(config.get(("dataset", "folder"), fallback=''))
        stimuli_path = os.path.expandvars(config.get(("dataset", "stimuli_path"), fallback=''))
        data_files = config.get(("dataset", "pre_processed"), fallback=None)
        preproc_pipeline = config.get(("dataset", "preprocess"), fallback="linear")
        target_sr = config.get(("dataset", "sr"), fallback=64)
        target_sr_aud = target_sr
        __cls__.selected_chs = np.array(config.get(("dataset", "channels")))
        if data_files is None:
            data_files = config.get(("dataset", "raw"), fallback=None)        
        all_sbjs = np.array(config.get(("dataset", "all_sbjs")))
        if subject is None:
            subjects = all_sbjs
        else:
            subjects = [subject]
        if __cls__.all_data is None:
            if preproc_pipeline == "ThorntonM":
                preproc = preprocessing_cnn.preprocess
            elif preproc_pipeline == "linear":
                preproc = preprocessing_linear.preprocess                
            else:
                preproc = preprocessing.preprocess       
            __cls__.all_data = []
            data_files = [os.path.join(data_folder, f) for f in data_files]
            for sbj in subjects:
                f = data_files[sbj]
                print(f'Loading data file {f}')
                s_data = []
                raw = loadmat(f, squeeze_me=True)['trials']
                __cls__.sr_eeg = raw[0]['FileHeader'].item()['SampleRate'].item()
                __cls__.channels = raw[0]['FileHeader'].item().item()[9]['Label'].tolist()
                __cls__.ordinary_selected_idx = [__cls__.channels.index(ch) for ch in __cls__.ordinary_channels]
                selected_idx = [__cls__.channels.index(ch) for ch in __cls__.selected_chs]
                for trial in range(len(raw)):
                    label = int(raw[trial]['attended_ear'].item()=='R')
                    stimuli = raw[trial]['stimuli'].item()
                    ref = raw[trial]['RawData'].item()['EegData'].item()[...,__cls__.ordinary_selected_idx].T.mean(axis=0, keepdims=True)
                    eeg = raw[trial]['RawData'].item()['EegData'].item()[...,selected_idx].T - ref
                    audio = []
                    sr_aud = []
                    for s in stimuli:
                        file = s.replace('_hrtf.wav', '_dry.wav')
                        sr, a = read(os.path.join(stimuli_path, file))
                        audio.append(a)
                        sr_aud.append(sr)
                    eeg,audio = preproc(eeg, audio, __cls__.sr_eeg, sr_aud, target_sr, target_sr_aud)
                    s_data.append({'eeg':eeg, 'audio':audio, 'label':label, 'stimuli':stimuli})
                    del eeg,audio
                del raw
                if subject is None:
                    __cls__.all_data.append(s_data)
                else:
                    __cls__.all_data = None 
                    return s_data
                    

    @classmethod
    def createSSCrossValidation(__cls__, subject, config: Config) -> list[tuple]:
        """Create train and test data for subject-specific validation"""
        preproc_pipeline = config.get(("dataset", "preprocess"), fallback="linear")
        nFold = config.get(("learning", "nFold"), fallback=8)
        subject_data=__cls__.loadData(config, subject)
        #
        crossSSData = []
        n_trials = len(subject_data)
        kf = KFold(n_splits=nFold, random_state=subject, shuffle=True)
        trial_idx_splits = [(train,test) for train,test in kf.split(range(n_trials))]         
        for fold in range(nFold):
            train_idxs = trial_idx_splits[fold][0]
            test_idxs = trial_idx_splits[fold][1]
            test_attended_stimuli = []
            tr_eeg, tr_aud, tr_label = [], [], []
            te_eeg, te_aud, te_label = [], [], []
            for trial in test_idxs:
                label = subject_data[trial]['label']
                attd_stimuliname = subject_data[trial]['stimuli'][label].replace('_dry.wav','').replace('_hrtf.wav','').replace('rep_','')
                test_attended_stimuli.append(attd_stimuliname)
                te_eeg.append(subject_data[trial]['eeg'])
                te_aud.append(subject_data[trial]['audio'])
                te_label.append(label)
            for trial in train_idxs:
                label = subject_data[trial]['label']
                attd_stimuliname = subject_data[trial]['stimuli'][label].replace('_dry.wav','').replace('_hrtf.wav','')
                if any(i in attd_stimuliname for i in test_attended_stimuli):
                    continue
                else:                
                    tr_eeg.append(subject_data[trial]['eeg'])
                    tr_aud.append(subject_data[trial]['audio'])
                    tr_label.append(label)
            crossSSData.append(((tr_eeg, tr_aud, tr_label),(te_eeg, te_aud, te_label)))
        return crossSSData
        
    @classmethod
    def createSICrossValidation(__cls__, subject, config: Config) -> list[tuple]:
        """Create train and test data for subject-independent validation"""
        preproc_pipeline = config.get(("dataset", "preprocess"), fallback="linear")
        all_sbjs = np.array(config.get(("dataset", "all_sbjs")))
        nFold = config.get(("learning", "nFold"), fallback=8)            
        num_sbjs = len(all_sbjs)
        __cls__.loadData(config)
        crossSIData = []
        #
        tr_sbjs = np.delete(all_sbjs, np.where(all_sbjs==subject)[0])
        test_data = __cls__.all_data[subject]
        n_trials = len(test_data)
        kf = KFold(n_splits=nFold, random_state=subject, shuffle=True)
        trial_idx_splits = [(train,test) for train,test in kf.split(range(n_trials))]         
        for fold in range(nFold):
            test_idxs = trial_idx_splits[fold][1]
            test_attended_stimuli = []
            tr_eeg, tr_aud, tr_label = [], [], []
            te_eeg, te_aud, te_label = [], [], []
            for trial in test_idxs:
                label = test_data[trial]['label']
                attd_stimuliname = test_data[trial]['stimuli'][label].replace('_dry.wav','').replace('_hrtf.wav','').replace('rep_','')
                test_attended_stimuli.append(attd_stimuliname)
                te_eeg.append(test_data[trial]['eeg'])
                te_aud.append(test_data[trial]['audio'])
                te_label.append(label)
            for tr_s in tr_sbjs:
                s_data = __cls__.all_data[tr_s]
                for trial in range(len(s_data)):
                    label = s_data[trial]['label']
                    attd_stimuliname = s_data[trial]['stimuli'][label].replace('_dry.wav','').replace('_hrtf.wav','')
                    if any(i in attd_stimuliname for i in test_attended_stimuli):
                        continue
                    else:
                        tr_eeg.append(s_data[trial]['eeg'])
                        tr_aud.append(s_data[trial]['audio'])
                        tr_label.append(label)
            crossSIData.append(((tr_eeg, tr_aud, tr_label),(te_eeg, te_aud, te_label)))
        return crossSIData        