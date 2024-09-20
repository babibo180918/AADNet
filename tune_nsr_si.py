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
N_TRIALS = 80

try:
     results_path = pathlib.Path('./output')
except KeyError:
     print('please configure the environment!')
     exit()

def setup_results_dir():

     Path(os.path.join(results_path, 'trained_models')).mkdir(parents=True, exist_ok=True)
     Path(os.path.join(results_path, 'predictions')).mkdir(parents=True, exist_ok=True)
     

def tune_dnns(load=False, models=['cnn', 'fcnn'], dataset='hugo', config=None, nFold=1):
    
    print(f'Running hyperparameter tuning for NSR with {dataset} in {nFold} folds')
    if 'cnn' in models:
        def cnn_objective(trial):
            train_params = {
                'lr': trial.suggest_categorical('tr_lr', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
                'batch_size':trial.suggest_categorical('tr_batch_size', [64, 128, 256]),
                'weight_decay': trial.suggest_categorical('tr_weight_decay', [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
                }
            model_kwargs = {
                'dropout_rate':trial.suggest_uniform('dropout_rate', 0, 0.4),
                'F1':trial.suggest_categorical('F1', [2, 4, 8]),
                'D':trial.suggest_categorical('D', [2, 4, 8])
            }
            model_kwargs['F2'] = model_kwargs['D']*model_kwargs['F1']
            if dataset=='hugo':
                accuracy, _ = train_dnn(data_file, range(13), None, **train_params, epochs=EPOCHS, model_handle=CNN, **model_kwargs, early_stopping_patience=PATIENCE, optuna_trial=trial)
            else:
                kf = KFold(n_splits=8, random_state=0, shuffle=True)   
                accuracy = np.zeros(nFold)
                if dataset=='DTUDataset':
                    pooledData = DTUDataset.getPooledData(config)
                    trial_idx_splits = [(train,valid) for train,valid in kf.split(range(60))]
                elif dataset=='Das2019Dataset':
                    pooledData = Das2019Dataset.getPooledData(config)
                    trial_idx_splits = [(train,valid) for train,valid in kf.split(range(12))]  
                elif dataset=='EventAADDataset':
                    pooledData = EventAADDataset.getPooledData(config)
                    trial_idx_splits = [(train,valid) for train,valid in kf.split(range(40))]
                for fold in range(nFold):
                    accuracy[fold],_ = train_dnn(None, np.array(config.get(('dataset', 'all_sbjs'))), None, **train_params, epochs=EPOCHS, dataset=dataset, model_handle=CNN, train_parts=trial_idx_splits[fold][0], val_parts=trial_idx_splits[fold][1], **model_kwargs, early_stopping_patience=PATIENCE, workers=config.get(('learning', 'running', 'num_workers')), optuna_trial=trial, data=pooledData, num_input_channels = len(config.get(('dataset', 'channels'))))
                    print(f'Finished tuning fold {fold}: {accuracy}')
                accuracy = accuracy.mean()
            return accuracy

        if load and os.path.exists(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_study.pk')):
            cnn_study = pickle.load(open(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_study.pk'), 'rb'))
        else:
            cnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)
            cnn_study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.RandomSampler(seed=0),
                pruner=cnn_pruner
            )

        cnn_study.optimize(cnn_objective, n_trials=N_TRIALS)
        cnn_summary = cnn_study.trials_dataframe()
        cnn_summary.to_csv(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_param_search.csv'))

        cnn_best_params = cnn_study.best_trial.params
        cnn_best_model_kwargs = {k: cnn_best_params[k] for k in cnn_best_params if not k.startswith('tr_')}
        cnn_best_model_kwargs['F2'] = cnn_best_model_kwargs['F1']*cnn_best_model_kwargs['D']
        cnn_best_train_params = {k.replace('tr_', ''): cnn_best_params[k] for k in cnn_best_params if k.startswith('tr_')}
        json.dump(cnn_best_model_kwargs, open(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_mdl_kwargs.json'), 'w'))
        json.dump(cnn_best_train_params, open(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_train_params.json'), 'w'))

        pickle.dump(cnn_study, open(os.path.join(results_path, 'trained_models', f'{dataset}_cnn_study.pk'), 'wb'))

    if 'fcnn' in models:
        def fcnn_objective(trial):
            train_params = {
                'lr': trial.suggest_categorical('tr_lr', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
                'batch_size':trial.suggest_categorical('tr_batch_size', [64, 128, 256]),
                'weight_decay': trial.suggest_categorical('tr_weight_decay', [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
                }
            model_kwargs = {
                'num_hidden': trial.suggest_int('num_hidden', 1,4),
                'dropout_rate': trial.suggest_uniform('dropout_rate', 0, 0.5)}
            accuracy, _ = train_dnn(data_file, range(13), None, **train_params, epochs=20, model_handle=FCNN, **model_kwargs, early_stopping_patience=PATIENCE, optuna_trial=trial)
            return accuracy

        if load and os.path.exists(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_study.pk')):
            fcnn_study = pickle.load(open(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_study.pk'), 'rb'))
        else:
            fcnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)
            fcnn_study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.RandomSampler(seed=0),
                pruner=fcnn_pruner
            )

        fcnn_study.optimize(fcnn_objective, n_trials=N_TRIALS)
        fcnn_summary = fcnn_study.trials_dataframe()
        fcnn_summary.to_csv(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_param_search.csv'))

        fcnn_best_params = fcnn_study.best_trial.params
        fcnn_best_model_kwargs = {k: fcnn_best_params[k] for k in fcnn_best_params if not k.startswith('tr_')}
        fcnn_best_train_params = {k.replace('tr_', ''): fcnn_best_params[k] for k in fcnn_best_params if k.startswith('tr_')}
        json.dump(fcnn_best_model_kwargs, open(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_mdl_kwargs.json'), 'w'))
        json.dump(fcnn_best_train_params, open(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_train_params.json'), 'w'))

        pickle.dump(fcnn_study, open(os.path.join(results_path, 'trained_models', f'{dataset}_fcnn_study.pk'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Tune hyperparameter for the CNN model.')
    parser.add_argument("-j", "--jobname", type=str, required=True, help="Name of running entity.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file path.")
    args = parser.parse_args()
    jobname = args.jobname
    config = Config.load_config(os.path.abspath(args.config))
    ds = config.get(('dataset', 'name'))
    setup_results_dir()
    tune_dnns(load=False, models=['cnn'], dataset=ds, config=config, nFold=1)
    