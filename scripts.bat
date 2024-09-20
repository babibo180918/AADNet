@echo off

SET DATASET=W:\data& REM path to the dataset

python cross_validate_loso.py -c config/config_AADNet_SI_Das.yml -j aadnet_si_das
python cross_validate_ss.py -c config/config_AADNet_SS_Das.yml -j aadnet_ss_das
python verify_channel_contribution.py -c config/config_AADNet_channel_distribution_Das.yml -j aadnet_channels_das
python tune_nsr_si.py -c config/config_NSR_SI_Das.yml -j tune_nsr_si
python tune_nsr_ss.py -c config/config_NSR_SS_Das.yml -j tune_nsr_ss

pause
