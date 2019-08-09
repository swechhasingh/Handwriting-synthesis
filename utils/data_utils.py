import numpy as np
from collections import Counter


def train_offset_normalization(data):
    """
       The co-ordinate offsets are normalised to
       mean 0, std. dev. 1 over the training set.
    """
    mean = data[:, :, 1:].mean(axis=(0, 1))
    data[:, :, 1:] -= mean
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
    data[:, :, 1:] *= std
    data[:, :, 1:] += mean

    return data


def data_normalization(data):
    """
       Data denormalization using train set mean and std
    """
    mean = data[:, 1:].mean(axis=0)
    data[:, 1:] -= mean
    std = data[:, 1:].std(axis=0)
    data[:, 1:] /= std

    return mean, std, data


def data_processing(data):
    """
       Data denormalization using train set mean and std
    """
    min_xy = data[:, 1:].min(axis=0)
    data[:, 1:] -= min_xy
    max_xy = data[:, 1:].max(axis=0)
    data[:, 1:] /= (max_xy - min_xy)
    data[:, 1:] *= 10
    return data
