# -*- encoding: utf-8 -*-
'''
@File        :_ree.py
@Time        :2021/04/05 13:14:10
@Author      :wlgls
@Version     :1.0
'''

import pywt
import numpy as np




def wavelet_features(data):
    """Time-Frequency feature. It is based on 《Classification of human emotion from EEG using discrete wavelet transform》.
    In this function, we use "db4" mother wavelet to decompose the signal into 4 layers.
    Warning: maybe It can only be used in deap datasets.

    Parameters
    ----------
    data : array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points) 

    Returns
    -------
    f:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)

    Examples:
    In [6]: data.shape, label.shape
    Out[6]: ((40, 32, 8064), (40, 1))

    In [7]: rwe(data).shape
    Out[7]: (40, 32, 5)
    """
    cA4, cD4, cD3, cD2, cD1  = pywt.wavedec(data, "db4", level=4)

    # waveEngergy
    p_cA4 = np.sum(cA4**2, axis=-1)
    p_cD4 = np.sum(cD4**2, axis=-1)
    p_cD3 = np.sum(cD3**2, axis=-1)
    p_cD2 = np.sum(cD2**2, axis=-1)
    p_cD1 = np.sum(cD1**2, axis=-1)
    

    p_sum = p_cA4 + p_cD4 + p_cD3 + p_cD2 + p_cD1

    # relative_wavelet_energy
    rp1 = p_cA4 / p_sum
    rp2 = p_cD4 / p_sum
    rp3 = p_cD3 / p_sum
    rp4 = p_cD2 / p_sum
    rp5 = p_cD1 / p_sum

    

    f = np.stack((rp1, rp2, rp3, rp4, rp5), axis=-1)
    entropy = np.sum(- f * np.log2(f), axis=-1)[..., np.newaxis]
    f = np.concatenate((f, entropy), axis=-1)
    return f




