import numpy as np


def get_strokes_length(strokes):
    """
       Returns length of sequences in dataset
    """
    lengths = []
    for stroke in strokes:
        lengths.append(stroke.shape[0])
    return lengths


def get_data_and_mask(strokes):
    """
       Converts list of different length sequences into a 3D array of size(n_total, max_len, 3),
       computes a mask used for computing loss.
       returns :
            data: size(n_total, max_len, 3)
            mask: size(n_total, max_len)
    """
    lengths = get_strokes_length(strokes)
    max_len = np.max(lengths)
    n_total = len(strokes)
    mask_shape = (n_total, max_len)
    mask = np.zeros(mask_shape, dtype=np.float32)
    data_shape = (n_total, max_len, 3)
    data = np.zeros(data_shape, dtype=np.float32)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.
        data[i, :length] = strokes[i]

    return data, mask


def get_inputs_and_targets(data):
    input_data = np.zeros(data.shape, dtype=np.float32)
    input_data[:, 1:, :] = data[:, :-1, :]
    target_data = data
    return input_data, target_data


def train_offset_normalization(data):
    """
       The co-ordinate offsets are normalised to
       mean 0, std. dev. 1 over the training set.
    """
    mean = data[:, :, 1:].mean(axis=(0, 1))
    datadata[:, :, 1:] -= mean
    std = data[:, :, 1:].std(axis=(0, 1))
    data[:, :, 1:] /= std

    return mean, std, data


def valid_offset_normalization(mean, std, data):
    """
       Data normalization using train set mean and std
    """
    data[:, :, 1:] -= mean
    data[:, :, 1:] /= std
    return data


def data_denormalization(mean, std, data):
    """
       Data denormalization using train set mean and std
    """
    data[:, : 1:] *= std
    data[:, : 1:] += mean

    return data
