
import numpy as np

def intercept_signal(signal, start, stop=None, sf=128):
    """Intercept the required signal

    Parameters
    ----------
    signal : ndarray
        Target data, only the last dimension will be operated.

    start : int
        Start of interval. If stop is not given, start defaults to 0.

    stop : int
        End of interval.

    sf : int, default=128
        Sampling frequency of signal.

    Return
    ---------
    data : ndarray
    """
    if stop is None:
        start, stop = 0, start

    point_of_start, point_of_end = start*sf, stop*sf

    return signal[..., point_of_start:point_of_end]

def split_signal(signal, labels=None, window=1, step=None,  sf=128, groupby='trial', shuffle=False):
    """Split a signal into small piece
    Warning : It's applicability is a little poor, It can only be used for data with shape like (video/trial,channel,point)
    Parameters
    ----------
    signal : ndarray
        Target data, only the last dimension will be operated.

    labels : ndarray, default=None
        Optional. Expand the label corresponding to the signal to the label corresponding to the pieces.

    window : int, default=1
        Window size of segmentation. In units of seconds.

    step : int, default=None

    sf : int, default=128
        Sampling frequency of signal.

    groups : str, default='slice'
        Group division of data，
        'trial' treats all slices of a trial as a group
        'slice' treats all trails of a slice as a group
        'all' will return both of them.
    Return
    ----------
    data : ndarray
    target : ndarray
        If labels if given.
    """

    if len(signal.shape) != 3:
        raise ValueError("Maybe it can't be used for this data. ")

    if step is None:
        step = window

    data = []

    start, stop, step = 0, int(window*sf), int(step*sf)
    while stop <= signal.shape[-1] or start == 0:
        data.append(signal[..., start:stop])
        start += step
        stop += step

    data = np.stack(data, axis=-3)

    trials, slices, *features_shape = data.shape
    data = data.reshape(-1, *features_shape)

    if groupby == 'trial':
        groups = np.arange(1, trials+1)
        groups = np.repeat(groups, slices)
    elif groupby == 'slice':
        groups = np.arange(1, slices+1)
        groups = np.tile(groups, trials)
    else:
        groups = np.zeros((trials*slices, 2))
        group_trail = np.arange(1, trials+1)
        group_trail = np.repeat(group_trail, slices)
        group_slice = np.arange(1, slices+1)
        group_slice = np.tile(group_slice, trials)
        groups[:, 0] = group_trail
        groups[:, 1] = group_slice

    if labels is not None:
        label = np.repeat(labels, slices, axis=0)
        # print(label.shape)
        # label = label.reshape(-1, label.shape[-1])
        if shuffle:
            index = np.arange(len(groups))
            np.random.shuffle(index)
            return groups[index], data[index], label[index]
        return groups,  data,  label

    return groups, data, None
