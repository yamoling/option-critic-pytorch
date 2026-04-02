from typing import Any
from marlenv import Observation, Episode, Transition
import torch
from torch import Tensor
from experience_replay import ReplayBuffer
from networks import OptionCritic
from copy import deepcopy


class OptionCriticTrainer:
    def __init__(
        self,
        oc: OptionCritic,
        device: torch.device,
        buffer_capacity: int,
        lr: float,
        gamma: float,
        entropy_regularizer: float,
        termination_regularizer: float,
        critic_update_interval: int,
        batch_size: int,
        freeze_interval: int,
        n_agents: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.device = device
        self.testing = False
        self.oc = oc.to(device).train(True)
        self.target_oc = deepcopy(oc).to(device).train(False).eval()

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.gamma = gamma
        self.entropy_reg = entropy_regularizer
        self.termination_reg = termination_regularizer
        self.optimizer = torch.optim.RMSprop(self.oc.parameters(), lr=lr)
        self.critic_update_interval = critic_update_interval
        self.batch_size = batch_size
        self.freeze_interval = freeze_interval

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        self.buffer.push(
            transition.obs.data,
            transition.obs.extras,
            transition["options"],
            transition.reward,
            transition.next_obs.data,
            transition.next_obs.extras,
            transition.done,
        )
        if len(self.buffer) < self.batch_size:
            return {}
        loss = self.actor_loss(
            transition.obs,
            transition["options"],
            transition["logp"],
            transition["entropy"],
            transition.reward.item(),
            transition.done,
            transition.next_obs,
        )
        if time_step % self.critic_update_interval == 0:
            loss += self.critic_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if time_step % self.freeze_interval == 0:
            self.target_oc.load_state_dict(self.oc.state_dict())
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        return {}

    def compute_next_state_value(self, next_obs: Tensor, next_extras: Tensor, options: Tensor) -> Tensor:
        next_termination_probs = self.oc.get_termination_probability(next_obs, next_extras, options)
        next_q_options = self.target_oc.compute_q_options(next_obs, next_extras)
        next_continued_q_option = torch.gather(next_q_options, -1, options.unsqueeze(-1)).squeeze(-1)
        next_best_q_option = next_q_options.max(dim=-1).values
        # Either we follow the same option or we take the max option.
        # -> Weighted sum of continuing the same option or following the best option.
        next_state_value = (1 - next_termination_probs) * next_continued_q_option + next_termination_probs * next_best_q_option
        # Mixing
        next_state_value = next_state_value.sum(dim=-1)
        return next_state_value

    def actor_loss(
        self,
        observation: Observation,
        agents_options: list[int],
        logp: Tensor,
        entropy: Tensor,
        reward: float,
        done: bool,
        next_observation: Observation,
    ) -> Tensor:
        obs, extras = observation.as_tensors(self.device)
        next_obs, next_extras = next_observation.as_tensors(self.device)
        options = torch.tensor(agents_options, device=self.device, dtype=torch.long)
        with torch.no_grad():
            next_state_value = self.compute_next_state_value(next_obs, next_extras, options)
            q_options = self.oc.compute_q_options(obs, extras)

        continued_q_option = q_options[range(self.n_agents), options]
        best_q_option = q_options.max(dim=-1).values

        # Estimated state value
        v = reward + (1 - done) * self.gamma * next_state_value
        # Option advantage
        adv = continued_q_option.sum(dim=-1) - v
        # Termination loss
        termination_probs = self.oc.get_termination_probability(obs, extras, options)
        termination_loss = termination_probs * (continued_q_option - best_q_option + self.termination_reg) * (1 - done)

        # actor-critic policy gradient with entropy regularization
        policy_loss = -logp * adv - self.entropy_reg * entropy
        actor_loss = termination_loss + policy_loss
        return actor_loss.sum()

    def critic_loss(self):
        obs, extras, options, rewards, next_obs, next_extras, dones = self.buffer.sample(self.batch_size)
        options = options.to(self.device)
        with torch.no_grad():
            next_states_values = self.compute_next_state_value(
                next_obs.to(self.device),
                next_extras.to(self.device),
                options,
            )
        targets = rewards.to(self.device) + (1 - dones.to(self.device)) * self.gamma * next_states_values

        # The loss is the TD loss of Q and the update target, so we need to calculate Q
        # Shape (batch_size, n_agents, ...)
        q_options = self.oc.compute_q_options(obs.to(self.device), extras.to(self.device))
        q_options = q_options.gather(-1, options.unsqueeze(-1)).squeeze(-1)
        # q_options = q_options[batch_idx, :, options]
        q_options = q_options.sum(-1)

        # to update Q we want to use the actual network, not the prime
        assert q_options.shape == targets.shape
        return torch.nn.functional.mse_loss(q_options, targets)

    def make_agent(self):
        from option_agent import OptionAgent

        return OptionAgent(self.oc.n_options, self.n_agents, self.oc, self.device)
