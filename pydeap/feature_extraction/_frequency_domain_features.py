# -*- encoding: utf-8 -*-
'''
@File        :_frequency_domain_features.py
@Time        :2021/04/16 20:08:47
@Author      :wlgls
@Version     :1.0
'''

import numpy as np
from scipy import signal

def bin_power(data, band=(4, 8, 12, 16, 25, 45), fs=128, combined=True):
    # 重构 批量化处理
    C = np.fft.fft(data)
    C = np.abs(C)
    power = []
    for b_index in range(len(band)-1):
        freq_s = float(band[b_index])
        freq_e = float(band[b_index+1])
        # 切分开始和结束
        start = int(np.floor(freq_s/fs*data.shape[-1]))
        end = int(np.floor(freq_e/fs*data.shape[-1]))
        power.append(np.sum(C[..., start:end], axis=-1))

    f = np.stack(power, axis=-1)
    if combined:
        f = f.reshape((*f.shape[:-2]))

    return f

def power_spectral_density(data, sf=128, nperseg=128, band=(4, 8, 14, 31, 65), combined=True):
    """The power of each frequency band is calculated according to the frequency band division，and then it combines the frequency band power into a feature vector. It mainly uses Welch method.
    
    Parameters
    ----------
    data : array
        data, for DEAP dataset, It's shape may be (n_trials,n_slices, n_channels, points) （40, 63, 32, 128）
    sf : int, optional
        sampling frequency, by default 128
    nperseg : int, optional
        for Welch method, According to scipy.signal.welch , by default 1
    band : tuple, optional
        boundary frequencies of bands, by default (4, 8, 14, 31, 65)
        e.g. for (4, 8, 14, 31, 65), It will calculate the power spectrum of theta(4~7Hz),alpha(8~13Hz),beta(14~30Hz) and gamma(31~64Hz).

    Returns
    -------
    f:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials,n_slices n_channels, points), the f.shape is (n_trials,n_slices, n_channels, n_features)
    
    Example
    ------
    In [5]: d, l = load_deap(path, 0)

    In [6]: d.shape, l.shape
    Out[6]: ((40, 32, 8064), (40, 1))

    In [7]: psd(d).shape
    Out[7]: (40, 32, 5) # Each channel has 5 bands of average power
    
    In [12]: d, l = split_signal(d, l)

    In [13]: d.shape, l.shape
    Out[13]: ((40, 63, 32, 128), (40, 63))

    In [14]: psd(d).shape
    Out[14]: (40, 63, 32, 4)

    """
    band = np.array(band)

    freqs, power = signal.welch(data, sf, nperseg=nperseg)
    
    freqband = np.hsplit(freqs, band)[1:-1] # Remove the beginning and the end

    # Get the index of the corresponding frequency band
    pindex = []
    for fb in freqband:
        pindex.append(np.where(np.in1d(freqs, fb))[0])
    
    # Get features
    f = []
    for index in pindex:
        f.append(np.mean(power[..., index], axis=-1))
    
    f = np.stack(f, axis=-1)
    if combined:
        f = f.reshape((*f.shape[:-2]))

    return f
