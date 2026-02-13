from typing import Sequence
import torch
import torch.nn as nn
from torch import Tensor
from marlenv import Observation
from torch.distributions import Categorical, Bernoulli
from math import exp
from utils import to_tensor
from args import Args


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
    modules = []
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


class OptionCriticConvLLE(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        extras_size: int,
        num_actions: int,
        num_options: int,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=int(1e6),
        eps_test=0.05,
        device: torch.device | str = "cpu",
        testing=False,
    ):
        super().__init__()

        self.in_channels = input_shape[0]
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.cnn, n_features = make_cnn(input_shape, filters=[32, 64, 64], kernel_sizes=[3, 3, 3], strides=[1, 1, 1])
        self.features = nn.Sequential(nn.Linear(n_features + extras_size, 512), nn.ReLU())
        self.Q = nn.Linear(512, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(512, num_options)  # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        if data.ndim < 4:
            data = data.unsqueeze(0)
        features = self.cnn(data)
        features = torch.concat((features, extras), dim=-1)
        state = self.features(features)
        return state

    def get_Q(self, state: Tensor):
        return self.Q.forward(state)

    def predict_option_termination(self, state: Tensor, current_option: Tensor):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()

        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state: Tensor):
        return self.terminations.forward(state).sigmoid()

    def get_action(self, state: torch.Tensor, available_actions: torch.Tensor, option: int):
        logits = state.data @ self.options_W[option] + self.options_b[option]
        logits[~available_actions] = -torch.inf  # Mask unavailable actions
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.numpy(force=True), logp, entropy

    def greedy_option(self, state: Tensor):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    def epsilon(self, t: int):
        if t >= self.eps_decay:
            return self.eps_min
        eps = self.eps_start - (self.eps_start - self.eps_min) * (t / self.eps_decay)
        return eps


class OptionCriticConv(nn.Module):
    def __init__(
        self,
        in_features,
        num_actions,
        num_options,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=int(1e6),
        eps_test=0.05,
        device="cpu",
        testing=False,
    ):
        super(OptionCriticConv, self).__init__()

        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.magic_number = 7 * 7 * 64
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU(),
        )

        self.Q = nn.Linear(512, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(512, num_options)  # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()

        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_action(self, state, option):
        logits = state.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


class OptionCriticFeatures(nn.Module):
    def __init__(
        self,
        in_features,
        num_actions,
        num_options,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=int(1e6),
        eps_test=0.05,
        device="cpu",
        testing=False,
    ):
        super(OptionCriticFeatures, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.features = nn.Sequential(nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())

        self.Q = nn.Linear(64, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(64, num_options)  # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_action(self, state, option):
        logits = state.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


def critic_loss(model, model_prime, data_batch, args: Args):
    obs, extras, options, rewards, next_obs, next_extras, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options = torch.LongTensor(options).to(model.device)
    rewards = torch.FloatTensor(rewards).to(model.device)
    masks = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.get_state(obs, extras).squeeze(0)
    Q = model.get_Q(states)

    # the update target contains Q_next, but for stable learning we use prime network for this
    with torch.no_grad():
        next_states_prime = model_prime.get_state(next_obs, next_extras).squeeze(0)
        next_Q_prime = model_prime.get_Q(next_states_prime)  # detach?

        # Additionally, we need the beta probabilities of the next state
        next_states = model.get_state(next_obs, next_extras).squeeze(0)
        next_termination_probs = model.get_terminations(next_states).detach()
        next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * (
        (1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob * next_Q_prime.max(dim=-1)[0]
    )

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err


def actor_loss(
    obs: Tensor,
    extras: Tensor,
    option: Tensor,
    logp: Tensor,
    entropy: Tensor,
    reward: Tensor,
    done: Tensor,
    next_obs: Tensor,
    next_extras: Tensor,
    model,
    model_prime,
    args: Args,
) -> Tensor:
    state = model.get_state(obs, extras)
    next_state = model.get_state(next_obs, next_extras)
    next_state_prime = model_prime.get_state(next_obs, next_extras)

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * args.gamma * (
        (1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob * next_Q_prime.max(dim=-1)[0]
    )

    # The termination loss
    termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)

    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss
