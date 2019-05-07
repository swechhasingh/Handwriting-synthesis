import torch
import torch.nn as nn
import numpy as np

from torch.utils import data
from torch.utils.data import DataLoader
from torch.distributions import bernoulli, uniform
import torch.nn.functional as F

from models.models import UnconditionalLSTM
from utils import plot_stroke
from utils.dataset import HandwritingDataset
from utils.model_utils import compute_unconditional_loss, stable_softmax
from utils.data_utils import get_data_and_mask, get_inputs_and_targets, train_offset_normalization, valid_offset_normalization, data_denormalization
