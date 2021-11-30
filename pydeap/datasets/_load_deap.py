# -*- encoding: utf-8 -*-
'''
@File        :_load_data.py
@Time        :2021/09/17 17:18:57
@Author      :wlgls
@Version     :2.0
'''

import numpy as np
import pickle
from collections.abc import Iterable

def load_deap(path, channels='only eeg', labels=None):
    """Load and return the DEAP dataset.
    
    Data Description:
    http://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
    
    Parameters
    ----------
    path : str
        Path to the file loading trial of subject
    
    channels : str, list or None, default='only eeg'
        Determine the channels. If "only eeg", load EEG channel only. If None, load all channel.
        Of course, you can provide a list of channels you need.
        You can see all the channel names in ：http://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
        
    labels : list of str, default=None
        If None, load all the labels. If not None, list of label names to load.
        
    Return
    ----------
    data: ndarray
        The raw data.
    target : ndarray
        The target labels.
    """
    
    index_of_chan = _get_channel_index(channels)
    
    index_of_labels = _get_label_index(labels)
    
    with open(path, 'rb') as f:
        subject = pickle.load(f, encoding='latin1')
        
    data = subject['data']
    target = subject['labels']
    
    return data[:, index_of_chan], target[:, index_of_labels]


def _get_channel_index(channels):
    
    chan = np.array(['FP1', 'AF3', 'F3',  'F7',  'FC5', 'FC1', 'C3',  'T7',  'CP5', 'CP1',
                    'P3',  'P7',  'PO3', 'O1',  'Oz',  'Pz',  'Fp2', 'AF4', 'Fz',  'F4',
                    'F8',  'FC6', 'FC2', 'Cz',  'C4',  'T8',  'CP6', 'CP2', 'P4',  'P8',
                    'PO4', 'O2',  'hEOG','vEOG','zEMG','tEMG','GSR', 'Resp','Plet','Temp'])
    
    if channels is None:
        index_of_chan = np.arange(40)
    
    elif isinstance(channels, str) and channels == 'only eeg':
        index_of_chan = np.arange(32)
        
    elif isinstance(channels, list):
        channels = np.array(channels)
        err_chan = np.setdiff1d(channels, chan)
        
        if err_chan.size:
            raise ValueError("No {} channels".format(list(err_chan)))
            
        index_of_chan = np.in1d(chan, channels).nonzero()[0]
    
    else:
        raise ValueError("No channels selected")
    return index_of_chan

def _get_label_index(labels):
    
    lab = np.array(['valence', 'arousal', 'dominance', 'liking'])
    index_of_label = np.arange(4) 
    
    if isinstance(labels, Iterable):
        # if labels is not None
        index_of_label = np.in1d(lab, labels).nonzero()[0]
    
    return index_of_label
        