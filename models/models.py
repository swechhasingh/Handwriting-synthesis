import torch
import math
import torch.nn as nn


class HandWritingPredictionNet(nn.Module):

    def __init__(self, hidden_size=400, n_layers=3, output_size=121, input_size=3):
        super(HandWritingPredictionNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.LSTM_layers = nn.ModuleList()
        self.LSTM_layers.append(nn.LSTM(input_size, hidden_size, 1, batch_first=True))
        for i in range(n_layers - 1):
            self.LSTM_layers.append(nn.LSTM(input_size + hidden_size, hidden_size, 1, batch_first=True))

        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)

        self.init_weight()

    def forward(self, inputs, initial_hidden):
        hiddens = []
        hidden_cell_state = []  # list of tuple(hn,cn) for each layer
        output, hidden = self.LSTM_layers[0](inputs, (initial_hidden[0][0:1], initial_hidden[1][0:1]))
        hiddens.append(output)
        hidden_cell_state.append(hidden)
        for i in range(1, self.n_layers):
            inp = torch.cat((inputs, output), dim=2)
            output, hidden = self.LSTM_layers[i](inp, (initial_hidden[0][i:i + 1], initial_hidden[1][i:i + 1]))
            hiddens.append(output)
            hidden_cell_state.append(hidden)
        inp = torch.cat(hiddens, dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, hidden_cell_state

    def init_hidden(self, batch_size):
        initial_hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                          torch.zeros(self.n_layers, batch_size, self.hidden_size))
        return initial_hidden

    def init_weight(self):
        k = math.sqrt(1. / self.hidden_size)
        for param in self.LSTM_layers.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.output_layer.bias, 0.)


class HandWritingSynthesisNet(nn.Module):

    def __init__(self, hidden_size=400, n_layers=3, output_size=121, window_size=77):
        super(HandWritingSynthesisNet, self).__init__()
        self.vocab_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        K = 10

        self.lstm_1 = nn.LSTM(3 + window_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(3 + window_size + hidden_size, hidden_size, batch_first=True)
        self.lstm_3 = nn.LSTM(3 + window_size + hidden_size, hidden_size, batch_first=True)

        self.window_layer = nn.Linear(hidden, 3 * K)
        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)

    def init_hidden(self, batch_size):
        initial_hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                          torch.zeros(self.n_layers, batch_size, self.hidden_size))
        window_vector = torch.zeros(batch_size, self.vocab_size)
        return initial_hidden,

    def one_hot_encoding(self, text):
        encoding = text.new_zeros((len(text), self.vocab_size))
        encoding[torch.arange(len(text)), text] = 1.
        return encoding

    def forward(self, inputs, targets, text):
        return 2
