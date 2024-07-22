import traceback

import torch
import torch.nn as nn
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
import math
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
error_handler = logging.FileHandler(filename='errors.log')
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)


class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ci = nn.Parameter(torch.Tensor(hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_cf = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.W_co = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.W_xh = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 4))
        self.init_weights()

    def get_initial_state(self, batch_size):
        h0 = torch.zeros(batch_size, self.hidden_size)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def init_weights(self):
        sd = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-sd, sd)

    def set_weights(self, constant):
        for weight in self.parameters():
            weight.data = weight.data * 0 + constant

    def forward(self, x: Tensor, states: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        bs, seq_sz, _ = x.size()
        hidden_seq = torch.jit.annotate(List[Tensor], [])

        h_t, c_t = states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            x_with_h = torch.cat([x_t, h_t], dim=1)
            z = x_with_h @ self.W_xh

            z_i, z_f, z_c, z_o = z.chunk(4, 1)

            i_t = torch.sigmoid(z_i + c_t * self.W_ci + self.b_i)

            f_t = torch.sigmoid(z_f + c_t * self.W_cf + self.b_f)

            c_t = f_t * c_t + i_t * torch.tanh(z_c + self.b_c)

            o_t = torch.sigmoid(z_o + c_t * self.W_co + self.b_o)

            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t)

        hidden_seq = torch.stack(hidden_seq, dim=1)

        return hidden_seq, (h_t, c_t)


class SoftWindow(nn.Module):
    def __init__(self, input_size, num_components):
        super().__init__()

        self.alpha = nn.Linear(input_size, num_components)
        self.beta = nn.Linear(input_size, num_components)
        self.k = nn.Linear(input_size, num_components)

    def forward(self, x: Tensor, c: Tensor, prev_k: Tensor):
        """
        :param x: tensor of shape (batch_size, 1, input_size)
        :param c: tensor of shape (batch_size, num_characters, alphabet_size)
        :param prev_k: tensor of shape (batch_size, num_components)
        :return: phi (attention weights) of shape (batch_size, 1, num_characters), k of shape (batch_size, num_components)
        """

        x = x[:, 0]

        alpha = torch.exp(self.alpha(x))
        beta = torch.exp(self.beta(x))
        k_new = prev_k + torch.exp(self.k(x))

        batch_size, num_chars, _ = c.shape
        phi = self.compute_attention_weights(alpha, beta, k_new, num_chars)
        return phi, k_new

    def compute_attention_weights(self, alpha, beta, k, char_seq_size: int):
        alpha = alpha.unsqueeze(2).repeat(1, 1, char_seq_size)
        beta = beta.unsqueeze(2).repeat(1, 1, char_seq_size)
        k = k.unsqueeze(2).repeat(1, 1, char_seq_size)
        u = torch.arange(char_seq_size, device=alpha.device)

        densities = alpha * torch.exp(-beta * (k - u) ** 2)
        phi = densities.sum(dim=1).unsqueeze(1)
        return phi

    @staticmethod
    def matmul_3d(phi, c):
        return torch.bmm(phi, c)


class SynthesisNetwork(nn.Module):
    @classmethod
    def get_default_model(cls, alphabet_size, device, bias=None):
        return cls(3, 400, alphabet_size, device, bias=bias)

    def __init__(self, input_size, hidden_size, alphabet_size, device,
                 gaussian_components=10, output_mixtures=20, bias=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.alphabet_size = alphabet_size
        self.device = device
        self.gaussian_components = gaussian_components

        self.lstm1 = PeepholeLSTM(input_size + alphabet_size, hidden_size)
        self.window = SoftWindow(hidden_size, gaussian_components)
        self.lstm2 = PeepholeLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.lstm3 = PeepholeLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.mixture = MixtureDensityLayer(hidden_size * 3, output_mixtures)

    def forward(self, x, c, w, k, h1, c1, h2, c2, h3, c3, bias):
        hidden1 = (h1, c1)
        hidden2 = (h2, c2)
        hidden3 = (h3, c3)

        x_with_w = torch.cat([x, w], dim=-1)
        h1, hidden1 = self.lstm1(x_with_w, hidden1)

        phi, k = self.window(h1, c, k)

        w = self.window.matmul_3d(phi, c)

        mixture, hidden2, hidden3 = self.compute_mixture(x, h1, w, hidden2, hidden3, bias)
        mixture = self.squeeze(mixture)
        pi, mu, sd, ro, eos = mixture
        h1, c1 = hidden1
        h2, c2 = hidden2
        h3, c3 = hidden3
        return pi, mu, sd, ro, eos, w, k, h1, c1, h2, c2, h3, c3, phi

    def compute_mixture(self, x: Tensor, h1: Tensor, w1: Tensor,
                        hidden2: Tuple[Tensor, Tensor], hidden3: Tuple[Tensor, Tensor], bias):
        inputs = torch.cat([x, h1, w1], dim=-1)
        h2, hidden2 = self.lstm2(inputs, hidden2)

        inputs = torch.cat([x, h2, w1], dim=-1)
        h3, hidden3 = self.lstm3(inputs, hidden3)

        inputs = torch.cat([h1, h2, h3], dim=-1)

        return self.mixture(inputs, bias), hidden2, hidden3

    def get_initial_states(self, batch_size: int):
        h0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def get_all_initial_states(self, batch_size: int):
        hidden1 = self.get_initial_states(batch_size)
        hidden2 = self.get_initial_states(batch_size)
        hidden3 = self.get_initial_states(batch_size)
        return hidden1, hidden2, hidden3

    def get_initial_input(self):
        return torch.zeros(1, 3, device=self.device)

    def get_initial_window(self, batch_size: int):
        return torch.zeros(batch_size, 1, self.alphabet_size, device=self.device)

    def squeeze(self, mixture: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        pi, mu, sd, ro, eos = mixture
        return pi[0, 0], mu[0, 0], sd[0, 0], ro[0, 0], eos[0, 0]

    def unsqueeze(self, mixture: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        pi, mu, sd, ro, eos = mixture
        return pi.unsqueeze(0), mu.unsqueeze(0), sd.unsqueeze(0), ro.unsqueeze(0), eos.unsqueeze(0)


class PrimedSynthesisNetwork(SynthesisNetwork):
    def sample_primed(self, primed_x, c, s, steps=700):
        c = torch.cat([c, s], dim=1)
        c = c.to(self.device)

        batch_size, u, _ = c.shape

        primed_x = torch.cat([self.get_initial_input().unsqueeze(0), primed_x], dim=1)
        w = self.get_initial_window(batch_size)
        k = torch.zeros(batch_size, self.gaussian_components, device=self.device, dtype=torch.float32)

        hidden1, hidden2, hidden3 = self.get_all_initial_states(batch_size)

        priming_steps = primed_x.shape[1]

        for t in range(priming_steps):
            x = primed_x[:, t].unsqueeze(1)

            x_with_w = torch.cat([x, w], dim=-1)
            h1, hidden1 = self.lstm1(x_with_w, hidden1)

            phi, k = self.window(h1, c, k)

            w = self.window.matmul_3d(phi, c)

            mixture, hidden2, hidden3 = self.compute_mixture(x, h1, w, hidden2, hidden3)

        states = (hidden1, hidden2, hidden3)
        outputs, _ = self._sample_sequence(x, c, w, k, states, stochastic=True, steps=steps)

        return outputs


class MixtureDensityLayer(nn.Module):
    def __init__(self, input_size, num_components):
        super().__init__()

        self.num_components = num_components
        self.pi = nn.Linear(input_size, num_components)
        self.mu = nn.Linear(input_size, num_components * 2)
        self.sd = nn.Linear(input_size, num_components * 2)
        self.ro = nn.Linear(input_size, num_components)
        self.eos = nn.Linear(input_size, 1)

    def forward(self, x, bias):
        pi_hat = self.pi(x)
        sd_hat = self.sd(x)

        bias = bias[0]
        pi = F.softmax(pi_hat * (1 + bias), dim=-1)
        sd = torch.exp(sd_hat - bias)

        mu = self.mu(x)
        ro = torch.tanh(self.ro(x))
        eos = torch.sigmoid(self.eos(x))
        return pi, mu, sd, ro, eos


def get_mean_prediction(output, device, stochastic):
    pi, mu, sd, ro, eos = output

    num_components = len(pi)

    component = torch.multinomial(pi, 1).item()

    mu1 = mu[component]
    mu2 = mu[component + num_components]

    sd1 = sd[component]
    sd2 = sd[component + num_components]

    component_ro = ro[component]

    xy = torch.cat([mu[component:component + 1], mu[component + num_components:component + num_components + 1]])

    try:
        xy = sample_from_bivariate_mixture(mu1, mu2, sd1, sd2, component_ro)
    except Exception:
        logger.exception('Failed to sample from bi-variate normal distribution:')

    eos[eos > 0.5] = 1
    eos[eos <= 0.5] = 0

    return torch.cat([xy, eos], dim=0)


def sample_from_bivariate_mixture(mu1, mu2, sd1, sd2, ro):
    cov_x_y = ro * sd1 * sd2
    sigma = torch.tensor([[sd1 ** 2, cov_x_y], [cov_x_y, sd2 ** 2]])

    loc = torch.tensor([mu1.item(), mu2.item()])
    gmm = torch.distributions.MultivariateNormal(loc, sigma)

    v = gmm.sample()

    return v


def expand_dims(shape):
    res = 1
    for dim in shape:
        res *= dim
    return res


# todo: avoid using shape and view()
# todo: reimplement forward to work with 1 element sequences of batch size 1
