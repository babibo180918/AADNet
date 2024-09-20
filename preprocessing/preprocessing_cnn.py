import os
import numpy as np
from scipy.signal import resample, hilbert
from scipy.stats import zscore
from scipy.io import loadmat, wavfile
from mne.io import read_raw_brainvision
from mne.filter import filter_data, resample
import matplotlib.pyplot as plt

def process_stimuli(story_idx):
    '''
    hpf: the high-pass filter window to be used
    lpfs: a list of N low-pass filter windows
    srate: target sampling rate

    returns: N filtered and resampled windowed
    '''

    story_part_name = story_part_names[story_idx]
    srate0, stimulus = wavfile.read(audio_path(story_part_name))
    duration = len(stimulus)/srate0

    out = np.abs(hilbert(stimulus))
    out = filter_data(out, srate0, None, 50)
    out = resample(out, 125, srate0)

    return zscore(out), duration

def process_eeg(participant_idx, lengths):

    participant=participants[participant_idx]
    raw = read_raw_brainvision(eeg_filepath(participant), preload=True)
    raw = raw.drop_channels('Sound')

    raw.filter(None, upbe)
    raw.resample(125)
    raw.filter(lpbe, None)

    eeg = zscore(raw.get_data(), axis=1)

    EEG = []

    for j in story_parts:

        length = lengths[j]
        start_time = onsets[participant_idx, j]

        EEG.append(eeg[:, int(start_time*125):int(start_time*125)+length])
    
    return EEG
    
def preprocess(eeg, audio, sr_eeg, sr_aud, target_sr_eeg, target_sr_aud):
    upbe = 8.0
    lpbe = 0.5
    upbe_aud = 50
    eeg_out = filter_data(eeg, sr_eeg, None, upbe, verbose='CRITICAL')
    eeg_out = resample(eeg_out, target_sr_eeg, sr_eeg, verbose='CRITICAL')
    eeg_out = filter_data(eeg_out, target_sr_eeg, lpbe, None, verbose='CRITICAL')
    eeg_out = zscore(eeg_out, axis=-1)
    #
    min_len = eeg_out.shape[-1]/target_sr_eeg
    aud_out = []
    for i in range(len(audio)):
        a = np.abs(hilbert(audio[i]))
        a = filter_data(a, sr_aud[i], None, upbe_aud, verbose='CRITICAL')
        a = resample(a, target_sr_aud, sr_aud[i], verbose='CRITICAL')
        aud_out.append(zscore(a, axis=-1))
        min_len = min(min_len, a.shape[-1]/target_sr_aud)
    eeg_out = eeg_out[...,:int(min_len*target_sr_eeg)]
    aud_out = [a[...,:int(min_len*target_sr_aud)] for a in aud_out]
    return eeg_out, np.array(aud_out)