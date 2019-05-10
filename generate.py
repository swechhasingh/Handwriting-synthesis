import torch
import torch.nn as nn
import numpy as np
import argparse

from torch.utils import data
from torch.utils.data import DataLoader
from torch.distributions import bernoulli, uniform
import torch.nn.functional as F

from models.models import HandWritingPredictionNet
from utils import plot_stroke
from utils.constants import Global
from utils.dataset import HandwritingDataset
from utils.model_utils import compute_unconditional_loss, stable_softmax
from utils.data_utils import train_offset_normalization, valid_offset_normalization, data_denormalization


def argparser():

    parser = argparse.ArgumentParser(description='PyTorch Handwriting Synthesis Model')
    parser.add_argument('--model', type=str, default='prediction')
    parser.add_argument('--model_path', type=str, default='./trainedModels/best_model.pt')
    parser.add_argument('--seq_len', type=int, default=700)
    parser.add_argument('--char_seq', type=str, default='This is real handwriting')
    parser.add_argument('--seed', type=int, default=212, help='random seed')
    parser.add_argument('--data_path', type=str, default='./data/')
    args = parser.parse_args()

    return args


def sample_from_out_dist(y_hat, bias=0.0):
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=0)

    eos_prob = F.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=0)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = F.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1)

    mu_k = y_hat.new_zeros(2)

    mu_k[0] = mu_1[K]
    mu_k[1] = mu_2[K]
    cov = y_hat.new_zeros(2, 2)
    cov[0, 0] = std_1[K].pow(2)
    cov[1, 1] = std_2[K].pow(2)
    cov[0, 1], cov[1, 0] = correlations[K] * std_1[K] * std_2[K], correlations[K] * std_1[K] * std_2[K]

    x = torch.normal(mean=torch.Tensor([0., 0.]), std=torch.Tensor([1., 1.])).to(device)
    Z = mu_k + torch.mv(cov, x)

    sample = y_hat.new_zeros(1, 1, 3)
    sample[0, 0, 0] = eos_sample.item()
    sample[0, 0, 1:] = Z
    return sample


def generate_unconditional_seq(model_path, seq_len, device):

    model = HandWritingPredictionNet()
    # load the best model
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # initial input
    inp = torch.zeros(1, 1, 3)
    p = uniform.Uniform(torch.tensor([-0.5, -0.5]), torch.tensor([0.5, 0.5]))
    co_offset = p.sample()
    inp[0, 0, 1:] = co_offset
    inp = inp.to(device)

    print("Input: ", inp)

    gen_seq = []
    batch_size = 1

    initial_hidden = model.init_hidden(batch_size)
    hidden = tuple([h.to(device) for h in initial_hidden])

    print("Generating sequence....")
    with torch.no_grad():
        for i in range(seq_len):

            y_hat, state = model.forward(inp, hidden)

            _hidden = torch.cat([s[0] for s in state], dim=0)
            _cell = torch.cat([s[1] for s in state], dim=0)
            hidden = (_hidden, _cell)

            y_hat = y_hat.squeeze()

            Z = sample_from_out_dist(y_hat)
            inp = Z
            gen_seq.append(Z)

    gen_seq = torch.cat(gen_seq, dim=1)
    gen_seq = gen_seq.detach().cpu().numpy()

    return gen_seq


def generate_conditional_sequence(model_path, char_seq, device):
    gen_seq = []
    return gen_seq

if __name__ == '__main__':

    args = argparser()

    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    model = args.model
    train_dataset = HandwritingDataset(args.data_path, split='train')
    if model == 'prediction':
        gen_seq = generate_unconditional_seq(model_path, args.seq_len, device)

    elif model == 'synthesis':
        gen_seq = generate_conditional_sequence(model_path, args.char_seq, device)

    # denormalize the generated offsets using train set mean and std
    gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)

    # plot the sequence
    plot_stroke(gen_seq[0], save_name="gen_seq.png")
