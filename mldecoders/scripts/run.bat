REM #!/bin/bash

REM source /home/mdt20/miniconda3/bin/activate env-dnns
REM cd W:\data\Das2019

REM export MLDECODERS_RESULTS_DIR="results/0.5-8Hz-090522"
REM export MLDECODERS_HUGO_DATA_FILE="data/hugo/0.5-8Hz/data.h5"
REM export MLDECODERS_OCTAVE_DATA_FILE="data/octave/0.5-8Hz/data.h5"
REM export MLDECODERS_EEG_UPBE="8"
REM export MLDECODERS_EEG_LPBE="0.5"

REM python data_preprocessing/process_hugo_data.py
REM python data_preprocessing/process_octave_data.py

REM python studies/train_hugo_population_models.py
REM python studies/train_hugo_subject_specific_models.py
REM python studies/train_hugo_leave_one_out_models.py 
REM python studies/train_octave_subject_specific_models.py
REM python studies/predict_octave_subject_specific_models.py

@echo off
cd D:\OneDrive\WORK\Coding\mldecoders
SET MLDECODERS_RESULTS_DIR="results/0.5-8Hz-090522"
SET MLDECODERS_DAS_DATA_FILE="data/das/0.5-8Hz/data.h5"
SET MLDECODERS_EEG_UPBE=8
SET MLDECODERS_EEG_LPBE=0.5

python data_preprocessing/process_das2019_data.py
REM python studies/train_das2019_leave_one_out_models.py 