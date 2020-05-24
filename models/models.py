import torch
import math
import torch.nn as nn
from torch.distributions import bernoulli, uniform
from utils.model_utils import stable_softmax


def sample_from_out_dist(y_hat, bias):
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=0)

    eos_prob = torch.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=0)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1)

    mu_k = y_hat.new_zeros(2)

    mu_k[0] = mu_1[K]
    mu_k[1] = mu_2[K]
    cov = y_hat.new_zeros(2, 2)
    cov[0, 0] = std_1[K].pow(2)
    cov[1, 1] = std_2[K].pow(2)
    cov[0, 1], cov[1, 0] = (
        correlations[K] * std_1[K] * std_2[K],
        correlations[K] * std_1[K] * std_2[K],
    )

    x = torch.normal(mean=torch.Tensor([0.0, 0.0]), std=torch.Tensor([1.0, 1.0])).to(
        y_hat.device
    )
    Z = mu_k + torch.mv(cov, x)

    sample = y_hat.new_zeros(1, 1, 3)
    sample[0, 0, 0] = eos_sample.item()
    sample[0, 0, 1:] = Z
    return sample


def sample_batch_from_out_dist(y_hat, bias):
    batch_size = y_hat.shape[0]
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=1)

    eos_prob = torch.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=1)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1).squeeze()

    mu_k = y_hat.new_zeros((y_hat.shape[0], 2))

    mu_k[:, 0] = mu_1[torch.arange(batch_size), K]
    mu_k[:, 1] = mu_2[torch.arange(batch_size), K]
    cov = y_hat.new_zeros(y_hat.shape[0], 2, 2)
    cov[:, 0, 0] = std_1[torch.arange(batch_size), K].pow(2)
    cov[:, 1, 1] = std_2[torch.arange(batch_size), K].pow(2)
    cov[:, 0, 1], cov[:, 1, 0] = (
        correlations[torch.arange(batch_size), K]
        * std_1[torch.arange(batch_size), K]
        * std_2[torch.arange(batch_size), K],
        correlations[torch.arange(batch_size), K]
        * std_1[torch.arange(batch_size), K]
        * std_2[torch.arange(batch_size), K],
    )

    X = torch.normal(
        mean=torch.zeros(batch_size, 2, 1), std=torch.ones(batch_size, 2, 1)
    ).to(y_hat.device)
    Z = mu_k + torch.matmul(cov, X).squeeze()

    sample = y_hat.new_zeros(batch_size, 1, 3)
    sample[:, 0, 0:1] = eos_sample
    sample[:, 0, 1:] = Z.squeeze()
    return sample


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
            self.LSTM_layers.append(
                nn.LSTM(input_size + hidden_size, hidden_size, 1, batch_first=True)
            )

        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)

        # self.init_weight()

    def forward(self, inputs, initial_hidden):
        hiddens = []
        hidden_cell_state = []  # list of tuple(hn,cn) for each layer
        output, hidden = self.LSTM_layers[0](
            inputs, (initial_hidden[0][0:1], initial_hidden[1][0:1])
        )
        hiddens.append(output)
        hidden_cell_state.append(hidden)
        for i in range(1, self.n_layers):
            inp = torch.cat((inputs, output), dim=2)
            output, hidden = self.LSTM_layers[i](
                inp, (initial_hidden[0][i: i + 1], initial_hidden[1][i: i + 1])
            )
            hiddens.append(output)
            hidden_cell_state.append(hidden)
        inp = torch.cat(hiddens, dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, hidden_cell_state

    def init_hidden(self, batch_size, device):
        initial_hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
        )
        return initial_hidden

    def init_weight(self):
        k = math.sqrt(1.0 / self.hidden_size)
        for param in self.LSTM_layers.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def generate(self, inp, hidden, seq_len, bias, style=None, prime=False):
        gen_seq = []

        with torch.no_grad():
            if prime:
                y_hat, state = self.forward(style, hidden)
                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)

                y_hat = y_hat[:, -1, :]
                y_hat = y_hat.squeeze()

                for i in range(style.shape[1]):
                    gen_seq.append(style[0:1, i: i + 1, :])

                Z = sample_from_out_dist(y_hat, bias)
                inp = Z
                gen_seq.append(Z)

            for i in range(seq_len):

                y_hat, state = self.forward(inp, hidden)

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)

                y_hat = y_hat.squeeze()

                Z = sample_from_out_dist(y_hat, bias)
                inp = Z
                gen_seq.append(Z)

        gen_seq = torch.cat(gen_seq, dim=1)
        gen_seq = gen_seq.detach().cpu().numpy()

        return gen_seq


class HandWritingSynthesisNet(nn.Module):

    def __init__(self, hidden_size=400, n_layers=3, output_size=121, window_size=77):
        super(HandWritingSynthesisNet, self).__init__()
        self.vocab_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        K = 10
        self.EOS = False
        self._phi = []

        self.lstm_1 = nn.LSTM(3 + self.vocab_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(
            3 + self.vocab_size + hidden_size, hidden_size, batch_first=True
        )
        # self.lstm_3 = nn.LSTM(
        #     3 + hidden_size, hidden_size, batch_first=True
        # )
        self.lstm_3 = nn.LSTM(
            3 + self.vocab_size + hidden_size, hidden_size, batch_first=True
        )

        self.window_layer = nn.Linear(hidden_size, 3 * K)
        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)
        # self.init_weight()

    def init_hidden(self, batch_size, device):
        initial_hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
        )
        window_vector = torch.zeros(batch_size, 1, self.vocab_size, device=device)
        kappa = torch.zeros(batch_size, 10, 1, device=device)
        return initial_hidden, window_vector, kappa

    def one_hot_encoding(self, text):
        N = text.shape[0]
        U = text.shape[1]
        encoding = text.new_zeros((N, U, self.vocab_size))
        for i in range(N):
            encoding[i, torch.arange(U), text[i].long()] = 1.0
        return encoding

    def compute_window_vector(self, mix_params, prev_kappa, text, text_mask, is_map):
        encoding = self.one_hot_encoding(text)
        mix_params = torch.exp(mix_params)

        alpha, beta, kappa = mix_params.split(10, dim=1)

        kappa = kappa + prev_kappa
        prev_kappa = kappa

        u = torch.arange(text.shape[1], dtype=torch.float32, device=text.device)

        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u).pow(2)), dim=1)
        if phi[0, -1] > torch.max(phi[0, :-1]):
            self.EOS = True
        phi = (phi * text_mask).unsqueeze(2)
        if is_map:
            self._phi.append(phi.squeeze(dim=2).unsqueeze(1))

        window_vec = torch.sum(phi * encoding, dim=1, keepdim=True)
        return window_vec, prev_kappa

    def init_weight(self):
        k = math.sqrt(1.0 / self.hidden_size)
        for param in self.lstm_1.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        for param in self.lstm_2.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        for param in self.lstm_3.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        nn.init.uniform_(self.window_layer.weight, a=-0.01, b=0.01)
        nn.init.constant_(self.window_layer.bias, 0.0)

        nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(
        self,
        inputs,
        text,
        text_mask,
        initial_hidden,
        prev_window_vec,
        prev_kappa,
        is_map=False,
    ):

        hid_1 = []
        window_vec = []

        state_1 = (initial_hidden[0][0:1], initial_hidden[1][0:1])

        for t in range(inputs.shape[1]):
            inp = torch.cat((inputs[:, t: t + 1, :], prev_window_vec), dim=2)

            hid_1_t, state_1 = self.lstm_1(inp, state_1)
            hid_1.append(hid_1_t)

            mix_params = self.window_layer(hid_1_t)
            window, kappa = self.compute_window_vector(
                mix_params.squeeze(dim=1).unsqueeze(2),
                prev_kappa,
                text,
                text_mask,
                is_map,
            )

            prev_window_vec = window
            prev_kappa = kappa
            window_vec.append(window)

        hid_1 = torch.cat(hid_1, dim=1)
        window_vec = torch.cat(window_vec, dim=1)

        inp = torch.cat((inputs, hid_1, window_vec), dim=2)
        state_2 = (initial_hidden[0][1:2], initial_hidden[1][1:2])

        hid_2, state_2 = self.lstm_2(inp, state_2)
        inp = torch.cat((inputs, hid_2, window_vec), dim=2)
        # inp = torch.cat((inputs, hid_2), dim=2)
        state_3 = (initial_hidden[0][2:], initial_hidden[1][2:])

        hid_3, state_3 = self.lstm_3(inp, state_3)

        inp = torch.cat([hid_1, hid_2, hid_3], dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, [state_1, state_2, state_3], window_vec, prev_kappa

    def generate(
        self,
        inp,
        text,
        text_mask,
        prime_text,
        prime_mask,
        hidden,
        window_vector,
        kappa,
        bias,
        is_map=False,
        prime=False,
    ):
        seq_len = 0
        gen_seq = []
        with torch.no_grad():
            batch_size = inp.shape[0]
            print("batch_size:", batch_size)
            if prime:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, prime_text, prime_mask, hidden, window_vector, kappa, is_map
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                # last time step hidden state
                hidden = (_hidden, _cell)
                # # last time step window vector
                # window_vector = window_vector[:, -1:, :]
                # # last time step output vector
                # y_hat = y_hat[:, -1, :]
                # # y_hat = y_hat.squeeze()
                # Z = sample_from_out_dist(y_hat, bias)
                # inp = Z
                # gen_seq.append(Z)
                self.EOS = False
                inp = inp.new_zeros(batch_size, 1, 3)
                _, window_vector, kappa = self.init_hidden(batch_size, inp.device)

            while not self.EOS and seq_len < 2000:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, text, text_mask, hidden, window_vector, kappa, is_map
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)
                # for batch sampling
                # y_hat = y_hat.squeeze(dim=1)
                # Z = sample_batch_from_out_dist(y_hat, bias)
                y_hat = y_hat.squeeze()
                Z = sample_from_out_dist(y_hat, bias)
                inp = Z
                gen_seq.append(Z)

                seq_len += 1

        gen_seq = torch.cat(gen_seq, dim=1)
        gen_seq = gen_seq.cpu().numpy()

        print("EOS:", self.EOS)
        print("seq_len:", seq_len)

        return gen_seq
