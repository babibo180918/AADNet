import sys
sys.path.append("./mldecoders")

import optuna
from torch.utils.data import DataLoader
from pipeline.training_functions import train_dnn, train_ridge
from pipeline.evaluation_functions import get_dnn_predictions, get_ground_truth
from pipeline.datasets import HugoMapped
from pipeline.dnn import CNN, FCNN
import pathlib
from pathlib import Path
import json
import os
import torch
import numpy as np
import pickle
import argparse

from aadnet.dataset import EventAADDataset, DTUDataset, Das2019Dataset
from utils.config import Config
from sklearn.model_selection import KFold

EPOCHS = 20
PATIENCE = 3
N_TRIALS = 5

try:
     results_path = pathlib.Path('./output')
except KeyError:
     print('please configure the environment!')
     exit()

def setup_results_dir():

     Path(os.path.join(results_path, 'trained_models')).mkdir(parents=True, exist_ok=True)
     Path(os.path.join(results_path, 'predictions')).mkdir(parents=True, exist_ok=True)
     
def tune_lrs(participant, models=['cnn', 'fcnn'], dataset='hugo', config=None, nFold=1):
    if 'cnn' in models:
        cnn_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_mdl_kwargs.json'), 'r'))
        cnn_train_params = json.load(open(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_train_params.json'), 'r'))
        
        if dataset in ['EventAADDataset', 'DTUDataset', 'Das2019Dataset']:
            sData = eval(config.get(('dataset', 'name'))).createSSCrossValidation(participant, config)
        
        for fold in range(nFold):
            del cnn_train_params['lr']
            if dataset in ['EventAADDataset', 'DTUDataset', 'Das2019Dataset']:
                data = sData[fold][0] # size of trial_numbers [trial_numbers][]
            def cnn_objective(trial):
                lr =  trial.suggest_loguniform('lr', 1e-8, 1e-1)
                print('>',lr)
                if dataset=='hugo':
                    accuracy, _ = train_dnn(data_file, participant, None, **cnn_train_params, lr=lr, epochs=EPOCHS, early_stopping_patience=PATIENCE,
                                        model_handle=CNN, **cnn_mdl_kwargs, optuna_trial=trial)
                else:
                    nInnerFold = 5
                    kf = KFold(n_splits=nInnerFold, random_state=0, shuffle=True)                
                    accuracy = np.zeros(nInnerFold)
                    nTrial = len(data[0])
                    trial_idx_splits = [(train,valid) for train,valid in kf.split(range(nTrial))]
                    for ifold in range(nInnerFold):
                        accuracy[ifold],_ = train_dnn(None, 0, None, **cnn_train_params, lr=lr, epochs=EPOCHS, dataset=dataset, model_handle=CNN, train_parts=trial_idx_splits[ifold][0], val_parts=trial_idx_splits[ifold][1], **cnn_mdl_kwargs, early_stopping_patience=PATIENCE, workers=config.get(('learning', 'running', 'num_workers')), optuna_trial=trial, data=[data], num_input_channels = len(config.get(('dataset', 'channels'))))
                        print(f'Finished tuning subject {participant} fold {fold}, ifold {ifold}: {accuracy[ifold]}')
                    accuracy = accuracy.mean()
                return accuracy

            gridsampler = optuna.samplers.GridSampler({"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]})

            cnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)
            cnn_study = optuna.create_study(
               direction="maximize",
               sampler=gridsampler,
               pruner=cnn_pruner   
            )

            cnn_study.optimize(cnn_objective, n_trials=N_TRIALS)
            cnn_summary = cnn_study.trials_dataframe()
            cnn_summary.to_csv(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_lr_search_P{participant:02d}_fold_{fold}.csv'))

            cnn_train_params['lr'] = cnn_study.best_trial.params['lr']
            json.dump(cnn_train_params, open(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_train_params_P{participant:02d}_fold_{fold}.json'), 'w'))

            pickle.dump(cnn_study, open(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_lr_study_P{participant:02d}_fold_{fold}.pk'), 'wb'))

    if 'fcnn' in models:
        fcnn_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_mdl_kwargs.json'), 'r'))
        fcnn_train_params = json.load(open(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_train_params.json'), 'r'))
        del fcnn_train_params['lr']

        def fcnn_objective(trial):
            lr =  trial.suggest_loguniform('lr', 1e-8, 1e-1)
            print('>',lr)
            accuracy, _ = train_dnn(data_file, participant, None, **fcnn_train_params, lr=lr, epochs=EPOCHS, early_stopping_patience=PATIENCE,
                                    model_handle=FCNN, **fcnn_mdl_kwargs, optuna_trial=trial)
            return accuracy

        gridsampler = optuna.samplers.GridSampler({"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]})
        fcnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)

        fcnn_study = optuna.create_study(
           direction="maximize",
           sampler=gridsampler,
           pruner=fcnn_pruner
        )
        fcnn_study.optimize(fcnn_objective, n_trials=N_TRIALS)
        fcnn_summary = fcnn_study.trials_dataframe()
        fcnn_summary.to_csv(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_lr_search_P{participant:02d}.csv'))

        fcnn_train_params['lr'] = fcnn_study.best_trial.params['lr']
        json.dump(fcnn_train_params, open(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_train_params_P{participant:02d}.json'), 'w'))

        pickle.dump(fcnn_study, open(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_lr_study_P{participant:02d}.pk'), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Tune hyperparameter for the SS CNN model.')
    parser.add_argument("-j", "--jobname", type=str, required=True, help="Name of running entity.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file path.")
    args = parser.parse_args()
    jobname = args.jobname
    config = Config.load_config(os.path.abspath(args.config))
    ds = config.get(('dataset', 'name'))
    setup_results_dir()
    from_sbj = config.get(('dataset', 'from_sbj'))
    to_sbj = config.get(('dataset', 'to_sbj'))
    for s in range(from_sbj, to_sbj):
        tune_lrs(participant=s, models=['cnn'], dataset=ds, config=config, nFold=8)
    