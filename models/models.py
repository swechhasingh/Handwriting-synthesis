import torch
import math
import torch.nn as nn


class UnconditionalLSTM(nn.Module):

    def __init__(self, hidden_size=900, n_layers=1, output_size=121, input_size=3):
        super(UnconditionalLSTM, self).__init__()
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
        hidden_cell_state = []  # list of (hn,cn)
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
        initial_hidden = (torch.Tensor(self.n_layers, batch_size, self.hidden_size),
                          torch.Tensor(self.n_layers, batch_size, self.hidden_size))
        return initial_hidden

    def init_weight(self):
        k = math.sqrt(1. / self.hidden_size)
        for param in self.LSTM_layers.parameters():
            nn.init.uniform_(param, a=-k, b=k)
            # if param.dim == 2:

            # elif param.dim == 1:
            #     nn.init.constant_(param, 0.)

        nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.output_layer.bias, 0.)
