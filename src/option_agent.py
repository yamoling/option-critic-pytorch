from marlenv import Observation
import torch
from torch.distributions import Bernoulli
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from option_critic import OptionCritic


class OptionAgent:
    def __init__(
        self,
        n_options: int,
        n_agents: int,
        oc: "OptionCritic",
        device: torch.device,
    ):
        self.current_options = [random.randint(0, n_options - 1) for _ in range(n_agents)]
        self.n_options = n_options
        self.oc = oc
        self._device = device
        self.epsilon = 0.1
        self.temperature = 1.0

    def _update_current_option(self, obs: torch.Tensor, extras: torch.Tensor):
        probs = self.oc.get_termination_probability(obs, extras, self.torch_options)
        terminated_options = Bernoulli(probs).sample().to(dtype=torch.bool).tolist()
        q_options = self.oc.compute_q_options(obs, extras)
        greedy_options = q_options.argmax(dim=-1).tolist()
        for i, (is_terminated, greedy_option) in enumerate(zip(terminated_options, greedy_options)):
            if not is_terminated:
                continue
            if random.random() < self.epsilon:
                self.current_options[i] = random.randint(0, self.n_options - 1)
            else:
                self.current_options[i] = greedy_option

    @property
    def torch_options(self):
        return torch.tensor(self.current_options, dtype=torch.long, device=self._device)

    def choose_action(self, observation: Observation):
        obs, extras = observation.as_tensors(self._device)
        self._update_current_option(obs, extras)
        action_dist = self.oc.policy(obs, extras, torch.from_numpy(observation.available_actions), self.torch_options)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.numpy(force=True), logp, entropy
