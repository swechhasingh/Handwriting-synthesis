import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.distributions import bernoulli, uniform
import torch.nn.functional as F

from models.models import HandWritingPredictionNet
from utils import plot_stroke
from utils.dataset import HandwritingDataset
from utils.model_utils import compute_unconditional_loss, stable_softmax
from utils.data_utils import train_offset_normalization, valid_offset_normalization, data_denormalization


def argparser():

    parser = argparse.ArgumentParser(description='PyTorch Handwriting Synthesis Model')
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='prediction')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--text_req', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=212, help='random seed')
    args = parser.parse_args()

    return args


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

        # Output gradient clipping
        y_hat.register_hook(lambda grad: torch.clamp(grad, -100, 100))

        loss.backward()

        # LSTM params gradient clipping
        nn.utils.clip_grad_value_(model.parameters(), 10)

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


def train(model, train_loader, valid_loader, batch_size, n_epochs, device):

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


if __name__ == "__main__":

    args = argparser()

    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = args.model
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    # Load the data and text
    train_dataset = HandwritingDataset(args.data_path, split='train', text_req=args.text_req)
    valid_dataset = HandwritingDataset(args.data_path, split='valid', text_req=args.text_req)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    if model == 'prediction':
        model = HandWritingPredictionNet(hidden_size=400, n_layers=3, output_size=121, input_size=3)
        model = train(model, train_loader, valid_loader, batch_size, n_epochs, device)
    elif model == 'synthesis':
        print("")
