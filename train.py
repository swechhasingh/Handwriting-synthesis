import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.distributions import bernoulli
import torch.nn.functional as F

from models.models import UnconditionalLSTM
from utils import plot_stroke
from utils.dataset import HandwritingDataset
from utils.model_utils import compute_unconditional_loss, stable_softmax
from utils.data_utils import get_data_and_mask, get_inputs_and_targets, train_offset_normalization, valid_offset_normalization, data_denormalization


def train_epoch(model, optimizer, epoch, train_loader, device):
    avg_loss = 0.0
    model.train()
    for i, (inputs, targets, mask) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        batch_size = inputs.shape[0]
        initial_hidden = model.init_hidden(batch_size)
        initial_hidden = tuple([h.to(device) for h in initial_hidden])
        optimizer.zero_grad()
        y_hat, state = model.forward(inputs, initial_hidden)
        loss = compute_unconditional_loss(targets, y_hat, mask)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        # print every 10 mini-batches
        if i % 10 == 0:
            print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, loss / batch_size))
    avg_loss /= len(train_loader.dataset)

    return avg_loss


def validation(model, valid_loader, device, epoch):
    avg_loss = 0.0
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, mask) in enumerate(valid_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            batch_size = inputs.shape[0]
            initial_hidden = model.init_hidden(batch_size)
            initial_hidden = tuple([h.to(device) for h in initial_hidden])
            y_hat, state = model.forward(inputs, initial_hidden)
            loss = compute_unconditional_loss(targets, y_hat, mask)
            avg_loss += loss.item()

            # print every 10 mini-batches
            if i % 10 == 0:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, loss / batch_size))

    avg_loss /= len(valid_loader.dataset)

    return avg_loss


def train(train_loader, valid_loader, batch_size, n_epochs, device):
    model = UnconditionalLSTM(hidden_size=400, n_layers=3, output_size=121, input_size=3)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    valid_losses = []

    for epoch in range(n_epochs):
        print("training.....")
        train_loss = train_epoch(model, optimizer, epoch, train_loader, device)
        print("validation....")
        valid_loss = validation(model, valid_loader, device, epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print('Epoch {}: Train: avg. loss: {:.3f}'.format(epoch + 1, train_loss))
        print('Epoch {}: Valid: avg. loss: {:.3f}'.format(epoch + 1, valid_loss))

    torch.save(model.state_dict(), "best_model.pt")

    return model


def sample_from_out_dist(y_hat):
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=0)

    eos_prob = F.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1], dim=0)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4])
    std_2 = torch.exp(y[5])
    correlations = F.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1)
    K = K.int().item()
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


def generate(model, seq_len, device):
    model.eval()
    inp = torch.zeros(1, 1, 3)
    inp = inp.to(device)

    gen_seq = []
    batch_size = inp.shape[0]
    initial_hidden = model.init_hidden(batch_size)
    hidden = tuple([h.to(device) for h in initial_hidden])

    for i in range(seq_len):

        y_hat, state = model.forward(inp, hidden)

        _hidden = torch.stack([s[0] for s in state], dim=0)
        _cell = torch.stack([s[1] for s in state], dim=0)
        hidden = (_hidden, _cell)

        y_hat = y_hat.squeeze()

        Z = sample_from_out_dist(y_hat)
        inp = Z
        gen_seq.append(Z.squeeze().cpu().numpy())

    gen_seq = np.array(gen_seq)
    plot_stroke(gen_seq)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    n_epochs = 1

    # Load the data and text
    strokes = np.load('./data/strokes.npy', allow_pickle=True, encoding='bytes')
    with open('./data/sentences.txt') as f:
        texts = f.readlines()

    data, mask = get_data_and_mask(strokes)

    idx_permute = np.random.permutation(data.shape[0])
    n_train = int(0.9 * data.shape[0])
    trainset = data[idx_permute[:n_train]]
    train_mask = mask[idx_permute[:n_train]]

    validset = data[idx_permute[n_train:]]
    valid_mask = mask[idx_permute[n_train:]]

    mean, std, normalized_trainset = train_offset_normalization(trainset)

    normalized_validset = valid_offset_normalization(mean, std, validset)

    train_inp, train_target = get_inputs_and_targets(normalized_trainset)
    valid_inp, valid_target = get_inputs_and_targets(normalized_validset)

    train_dataset = HandwritingDataset(train_inp, train_target, train_mask)
    valid_dataset = HandwritingDataset(valid_inp, valid_target, valid_mask)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = train(train_loader, valid_loader, batch_size, n_epochs, device)
    seq_len = 700
    generate(model, seq_len, device)
