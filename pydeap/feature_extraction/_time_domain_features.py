# -*- encoding: utf-8 -*-
'''
@File        :_time_domain_features.py
@Time        :2021/04/16 20:02:55
@Author      :wlgls
@Version     :1.0
'''

import numpy as np

def statistics(data, combined=True):
    """Statistical features， include Power, Mean, Std, 1st differece, Normalized 1st difference, 2nd difference,  Normalized 2nd difference.

    Parameters
    ----------
    data array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points)
    
    Return
    ----------
    f:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)

    Example
    ----------
    In [13]: d.shape, l.shape
    Out[13]: ((40, 32, 8064), (40, 1))

    In [14]: statistics_feature(d).shape
    Out[14]: (40, 32, 7)
    """
    # Power
    power = np.mean(data**2, axis=-1)
    # Mean
    ave = np.mean(data, axis=-1)
    # Standard Deviation
    std = np.std(data, axis=-1)
    # the mean of the absolute values of 1st differece mean
    diff_1st = np.mean(np.abs(np.diff(data,n=1, axis=-1)), axis=-1)
    # the mean of the absolute values of Normalized 1st difference
    normal_diff_1st = diff_1st / std
    # the mean of the absolute values of 2nd difference mean 
    diff_2nd = np.mean(np.abs(data[..., 2:] - data[..., :-2]), axis=-1)
    # the mean of the absolute values of Normalized 2nd difference
    normal_diff_2nd = diff_2nd / std
    # Features.append(np.concatenate((Power, Mean, Std, diff_1st, normal_diff_1st, diff_2nd, normal_diff_2nd), axis=2))
    
    f = np.stack((power, ave, std, diff_1st, normal_diff_1st, diff_2nd, normal_diff_2nd), axis=-1)
    if combined:
        f = f.reshape((*f.shape[:-2]))

    return f


def hjorth(data, combined=True):
    """Solving Hjorth features， include activity, mobility, complexity

    Parameters
    ----------
    data array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points)
    
    Return
    ----------
    f:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)

    Example
    ----------
    In [15]: d.shape, l.shape
    Out[15]: ((40, 32, 8064), (40, 1))

    In [16]: hjorth_features(d).shape
    Out[16]: (40, 32, 3)
    """
    data = np.array(data)
    ave = np.mean(data, axis=-1)[..., np.newaxis]
    diff_1st = np.diff(data, n=1, axis=-1)
    # print(diff_1st.shape)
    diff_2nd = data[..., 2:] - data[..., :-2]
    # Activity
    activity = np.mean((data-ave)**2, axis=-1)
    # print(Activity.shape)
    # Mobility
    varfdiff = np.var(diff_1st, axis=-1)
    # print(varfdiff.shape)
    mobility = np.sqrt(varfdiff / activity)

    # Complexity
    varsdiff = np.var(diff_2nd, axis=-1)
    complexity = np.sqrt(varsdiff/varfdiff) / mobility

    f = np.stack((activity, mobility, complexity), axis=-1)
    if combined:
        f = f.reshape((*f.shape[:-2]))
    return f


def higher_order_crossing(data, k=10, combined=True):
    """Solving the feature of hoc. Hoc is a high order zero crossing quantity.

    Parameters
    ----------
    data : array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points) 
    k : int, optional
        Order, by default 10
    
    Return
    ----------
    nzc:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)

    Example
    ----------
    In [4]: d, l = load_deap(path, 0)

    In [5]: hoc(d, k=10).shape
    Out[5]: (40, 32, 10)

    In [6]: hoc(d, k=5).shape
    Out[6]: (40, 32, 5)
    """
    nzc = []
    for i in range(k):
        curr_diff = np.diff(data, n=i)
        x_t = curr_diff >= 0
        x_t = np.diff(x_t)
        x_t = np.abs(x_t)

        count = np.count_nonzero(x_t, axis=-1)
        nzc.append(count)
    f = np.stack(nzc, axis=-1)
    if combined:
        f = f.reshape((*f.shape[:-2]))
    return f


def sevcik_fd(data, combined=True):
    """Fractal dimension feature is solved, which is used to describe the shape information of EEG time series data. It seems that this feature can be used to judge the electrooculogram and EEG.The calculation methods include Sevcik, fractal Brownian motion, box counting, Higuchi and so on.

    Sevcik method: fast calculation and robust analysis of noise
    Higuchi: closer to the theoretical value than box counting

    The Sevick method is used here because it is easier to implement
    Parameters
    ----------
    Parameters
    ----------
    data array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points)
    
    Return
    ----------
    f:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)
    
    Example
    ----------
    In [7]: d.shape, l.shape
    Out[7]: ((40, 32, 8064), (40, 1))

    In [8]: sevcik_fd(d).shape
    Out[8]: (40, 32, 1)

    """

    points = data.shape[-1]

    x = np.arange(1, points+1)
    x_ = x / np.max(x)

    miny = np.expand_dims(np.min(data, axis=-1), axis=-1)
    maxy = np.expand_dims(np.max(data, axis=-1), axis=-1)
    y_ = (data-miny) / (maxy-miny)

    L = np.expand_dims(np.sum(np.sqrt(np.diff(y_, axis=-1)**2 + np.diff(x_)**2), axis=-1), axis=-1)
    f = 1 + np.log(L) / np.log(2 * (points-1))
    # print(FD.shape)
    if combined:
        f = f.reshape((*f.shape[:-2]))
    return f

def calc_L(X, k, m):
    """
    Return Lm(k) as the length of the curve.
    """
    N = X.shape[-1]

    n = np.floor((N-m)/k).astype(np.int64)
    norm = (N-1) / (n*k)

    ss = np.sum(np.abs(np.diff(X[..., m::k], n=1)), axis=-1)

    Lm = (ss*norm) / k

    return Lm

def calc_L_average(X, k):
    """
    Return <L(k)> as the average value over k sets of Lm(k).
    """
    calc_L_series = np.frompyfunc(lambda m: calc_L(X, k, m), 1, 1)

    L_average = np.average(calc_L_series(np.arange(1, k+1)))

    return L_average

def higuchi_fd(data, k_max, combined=True):
    """Fractal dimension feature is solved, which is used to describe the shape information of EEG time series data. It seems that this feature can be used to judge the electrooculogram and EEG.The calculation methods include Sevcik, fractal Brownian motion, box counting, Higuchi and so on.

    Sevcik method: fast calculation and robust analysis of noise
    Higuchi: closer to the theoretical value than box counting

    The higuchi method is used here because it is easier to implement
    Parameters
    ----------
    Parameters
    ----------
    data array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points)
    
    Return
    ----------
    f:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)
    
    Example
    ----------
    In [7]: d.shape, l.shape
    Out[7]: ((40, 32, 8064), (40, 1))

    In [8]: higuchi_fd(dif combined:
        f = f

    return ).shape
    Out[8]: (40, 32, 1)
    """
    calc_L_average_series = np.frompyfunc(lambda k: calc_L_average(data, k), 1, 1)

    k = np.arange(1, k_max+1)

    L = calc_L_average_series(k)

    L = np.stack(L, axis=-1)
    fd = np.zeros(data.shape[:-1])
    for ind in np.argwhere(L[..., 0]):
        tmp = L[ind[0], ind[1], ind[2]]
        D, _= np.polyfit(np.log2(k), np.log2(tmp), 1)
        fd[ind[0], ind[1if combined:
        f = f

    return ], ind[2]] = - D
    f = np.expand_dims(fd, axis=-1)
    if combined:
        f = f.reshape((*f.shape[:-2]))
    return f
