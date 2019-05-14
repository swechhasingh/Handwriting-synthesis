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

    def init_hidden(self, batch_size, device):
        initial_hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                          torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))
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
        self.EOS = False

        self.lstm_1 = nn.LSTM(3 + self.vocab_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(3 + self.vocab_size + hidden_size, hidden_size, batch_first=True)
        self.lstm_3 = nn.LSTM(3 + self.vocab_size + hidden_size, hidden_size, batch_first=True)

        self.window_layer = nn.Linear(hidden_size, 3 * K)
        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)
        # self.init_weight()

    def init_hidden(self, batch_size, device):
        initial_hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                          torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))
        window_vector = torch.zeros(batch_size, 1, self.vocab_size, device=device)
        kappa = torch.zeros(batch_size, 10, 1, device=device)
        return initial_hidden, window_vector, kappa

    def one_hot_encoding(self, text):
        N = text.shape[0]
        U = text.shape[1]
        encoding = text.new_zeros((N, U, self.vocab_size))
        for i in range(N):
            encoding[i, torch.arange(U), text[i].long()] = 1.
        return encoding

    def compute_window_vector(self, mix_params, prev_kappa, text, text_mask):
        encoding = self.one_hot_encoding(text)
        mix_params = torch.exp(mix_params)
        alpha, beta, kappa = mix_params.split(10, dim=1)
        kappa = kappa + prev_kappa
        prev_kappa = kappa
        u = text.new_tensor(torch.arange(text.shape[1]), dtype=torch.float32)
        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u).pow(2)), dim=1)
        if phi[0, -1] > torch.max(phi[0, :-1]):
            self.EOS = True
            print(self.EOS)
        phi = (phi * text_mask).unsqueeze(2)
        window_vec = torch.sum(phi * encoding, dim=1, keepdim=True)
        return window_vec, prev_kappa

    def init_weight(self):
        k = math.sqrt(1. / self.hidden_size)
        for param in self.lstm_1.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        for param in self.lstm_2.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        for param in self.lstm_3.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        nn.init.uniform_(self.window_layer.weight, a=-0.01, b=0.01)
        nn.init.constant_(self.window_layer.bias, 0.)

        nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.output_layer.bias, 0.)

    def forward(self, inputs, text, text_mask, initial_hidden, prev_window_vec, prev_kappa):

        hid_1 = []
        window_vec = []

        state_1 = (initial_hidden[0][0:1], initial_hidden[1][0:1])

        for t in range(inputs.shape[1]):
            inp = torch.cat((inputs[:, t:t + 1, :], prev_window_vec), dim=2)

            hid_1_t, state_1 = self.lstm_1(inp, state_1)
            hid_1.append(hid_1_t)

            mix_params = self.window_layer(hid_1_t)
            window, kappa = self.compute_window_vector(mix_params.squeeze(dim=1).unsqueeze(2),
                                                       prev_kappa,
                                                       text,
                                                       text_mask)

            prev_window_vec = window
            prev_kappa = kappa
            window_vec.append(window)

        hid_1 = torch.cat(hid_1, dim=1)
        window_vec = torch.cat(window_vec, dim=1)

        inp = torch.cat((inputs, hid_1, window_vec), dim=2)
        state_2 = (initial_hidden[0][1:2], initial_hidden[1][1:2])

        hid_2, state_2 = self.lstm_2(inp, state_2)

        inp = torch.cat((inputs, hid_2, window_vec), dim=2)
        state_3 = (initial_hidden[0][2:], initial_hidden[1][2:])

        hid_3, state_3 = self.lstm_3(inp, state_3)

        inp = torch.cat([hid_1, hid_2, hid_3], dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, [state_1, state_2, state_3]
