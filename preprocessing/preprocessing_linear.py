import os
import numpy as np
from scipy.signal import resample, hilbert, gammatone, lfilter
from scipy.stats import zscore
from scipy.io import loadmat, wavfile
from mne.io import read_raw_brainvision
from mne.filter import filter_data, resample
import matplotlib.pyplot as plt

def plotAudio(audio, env):
    plt.clf()
    plt.plot(audio, 'r')
    plt.plot(env, 'b')
    plt.show()    

def getGMTFEnv(audio, sr, bp, target_sr_aud):
    '''
    extract the audio envelope using gammatone filterbank.
    '''
    srInt1 = 8000 # intermediate sampling frequency for auditory filterbank
    srInt2 = 128 # intermediate sampling frequency for filtering
    spacing = 1.5 # spacing auditory filterbank
    freqs = np.array([178.7, 250.3, 334.5, 433.5, 549.9, 686.8, 847.7, 1036.9, 1259.3, 1520.9, 1828.4, 2190.0, 2615.1, 3114.9, 3702.6])
    audio = filter_data(audio, sr, None, srInt1/2, verbose='CRITICAL')
    audio = resample(audio, srInt1, sr)
    envs = []
    for f in freqs:
        b,a =  gammatone(freq=f, ftype='fir', order=4, fs=srInt1)
        env = np.real(lfilter(b, a, audio))
        envs.append(np.abs(env)**0.6)
    envs = np.array(envs)
    envs = np.sum(envs, axis=0, keepdims=False)
    envs = resample(envs, srInt2, srInt1)
    envs = filter_data(envs, srInt2, bp[0], bp[1], verbose='CRITICAL')
    envs = resample(envs, target_sr_aud, srInt2)
    # plotAudio(resample(audio, target_sr_aud, srInt1), envs)
    return zscore(envs)
    
def preprocess(eeg, audio, sr_eeg, sr_aud, target_sr_eeg, target_sr_aud):
    upbe = 32.0
    lpbe = 0.5
    upbe_aud = 32.0
    lpbe_aud = 0.5
    eeg_out = eeg - eeg.mean(axis=0, keepdims=True) # rereference to the average
    eeg_out = filter_data(eeg_out, sr_eeg, None, upbe, verbose='CRITICAL')
    eeg_out = resample(eeg_out, target_sr_eeg, sr_eeg, verbose='CRITICAL')
    eeg_out = filter_data(eeg_out, target_sr_eeg, lpbe, None, verbose='CRITICAL')
    eeg_out = zscore(eeg_out, axis=-1)
    #
    min_len = eeg_out.shape[-1]/target_sr_eeg
    aud_out = []
    for i in range(len(audio)):
        a = getGMTFEnv(audio[i].astype(float), sr_aud[i], (lpbe_aud, upbe_aud), target_sr_aud)
        aud_out.append(a)
        min_len = min(min_len, a.shape[-1]/target_sr_aud)
    eeg_out = eeg_out[...,:int(min_len*target_sr_eeg)]
    aud_out = [a[...,:int(min_len*target_sr_aud)] for a in aud_out]
    return eeg_out, np.array(aud_out)