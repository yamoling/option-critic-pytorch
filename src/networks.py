import torch
import math
from torch import Tensor, nn
from typing import Sequence
from abc import ABC, abstractmethod
from marlenv import MARLEnv, Observation
from torch.distributions.categorical import Categorical


class OptionCritic(torch.nn.Module, ABC):
    def __init__(self, n_options: int):
        torch.nn.Module.__init__(self)
        ABC.__init__(self)
        self.n_options = n_options

    @abstractmethod
    def compute_q_options(self, obs: Tensor, extras: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_termination_probability(self, obs: Tensor, extras: Tensor, options: Tensor) -> Tensor:
        pass

    @abstractmethod
    def policy(self, obs: Tensor, extras: Tensor, available_actions: torch.Tensor, options: Tensor) -> torch.distributions.Categorical:
        """Compute the policy distribution for the given observation, extras and following the given options."""


class CNNOptionCritic(OptionCritic):
    def __init__(
        self,
        n_options: int,
        encoder: "CNNStateEncoder",
        policies: torch.nn.ModuleList,
        terminations: torch.nn.Module,
        q_option: torch.nn.Module,
    ):
        super().__init__(n_options)
        self.policies = policies
        self.terminations = terminations
        self.q_option = q_option
        self.state_encoder = encoder

    def get_termination_probability(self, obs: Tensor, extras: Tensor, options: Tensor) -> Tensor:
        state = self.state_encoder.batch_forward(obs, extras)
        x = self.terminations.forward(state)
        options = options.unsqueeze(-1)
        probs = torch.gather(x, -1, options)
        return probs.squeeze(-1)
        # options = options.flatten()
        # probs = x[range(batch_size), options]
        # return probs.view(*dims, -1)

    def compute_q_options(self, obs: Tensor, extras: Tensor) -> Tensor:
        state = self.state_encoder.batch_forward(obs, extras)
        return self.q_option.forward(state)

    def policy(self, obs: Tensor, extras: Tensor, available_actions: Tensor, options: Tensor) -> Categorical:
        states = self.state_encoder.batch_forward(obs, extras)
        logits_list = [self.policies[option].forward(s) for (s, option) in zip(states, options.tolist())]
        logits = torch.stack(logits_list)
        logits[~available_actions] = -torch.inf
        action_probs = torch.softmax(logits, dim=-1)
        action_dist = Categorical(action_probs)
        return action_dist


class VDN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, values: Tensor):
        return values.sum(-1)


class CNNStateEncoder(torch.nn.Module):
    def __init__(self, env: MARLEnv, device: torch.device):
        super().__init__()
        self.cnn, n_features = make_cnn(env.observation_shape, filters=[32, 64, 64], kernel_sizes=[3, 3, 3], strides=[1, 1, 1])
        self.features = nn.Sequential(
            nn.Linear(n_features + env.extras_shape[0], 512),
            nn.Tanh(),
        )
        self._device = device

    def forward(self, obs: Observation):
        data, extras = obs.as_tensors(self._device)
        return self.batch_forward(data, extras)

    def batch_forward(self, obs: Tensor, extras: Tensor):
        *dims, channels, height, width = obs.shape
        leading_dims = math.prod(dims)
        obs = obs.view(leading_dims, channels, height, width)
        x = self.cnn.forward(obs)
        extras = extras.view(leading_dims, -1)
        x = torch.concat([x, extras], dim=-1)
        x = self.features.forward(x)
        return x.view(*dims, -1)


def make_cnn(input_shape, filters: Sequence[int], kernel_sizes: Sequence[int], strides: Sequence[int], min_output_size=1024):
    """Create a CNN with flattened output based on the given filters, kernel sizes and strides."""
    channels, height, width = input_shape
    paddings = [0 for _ in filters]
    n_padded = 0
    output_w, output_h = conv2d_size_out(width, height, kernel_sizes, strides, paddings)
    output_size = filters[-1] * output_w * output_h
    while output_w <= 1 or output_h <= 1 or output_size < min_output_size:
        # Add paddings if the output size is negative
        paddings[n_padded % len(paddings)] += 1
        n_padded += 1
        output_w, output_h = conv2d_size_out(width, height, kernel_sizes, strides, paddings)
        output_size = filters[-1] * output_w * output_h
    assert output_h > 0 and output_w > 0, f"Input size = {input_shape}, output witdh = {output_w}, output height = {output_h}"
    modules = list[torch.nn.Module]()
    for f, k, s, p in zip(filters, kernel_sizes, strides, paddings):
        modules.append(torch.nn.Conv2d(in_channels=channels, out_channels=f, kernel_size=k, stride=s, padding=p))
        modules.append(torch.nn.ReLU())
        channels = f
    modules.append(torch.nn.Flatten())
    return torch.nn.Sequential(*modules), output_size


def conv2d_size_out(input_width: int, input_height: int, kernel_sizes: Sequence[int], strides: Sequence[int], paddings: Sequence[int]):
    """
    Compute the output width and height of a sequence of 2D convolutions.
    See shape section on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    width = input_width
    height = input_height
    for kernel_size, stride, pad in zip(kernel_sizes, strides, paddings):
        width = (width + 2 * pad - (kernel_size - 1) - 1) // stride + 1
        height = (height + 2 * pad - (kernel_size - 1) - 1) // stride + 1
    return width, height
