import torch
import torch.nn as nn


class UnconditionalLSTM(nn.Module):

    def __init__(self, input_size=3, hidden_size=900, output_size=121, n_layers=1):
        super(UnconditionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm_layers = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        return output
