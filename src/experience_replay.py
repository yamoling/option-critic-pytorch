import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, extras, option, reward, next_obs, next_extras, done):
        self.buffer.append((obs, extras, option, reward, next_obs, next_extras, done))

    def sample(self, batch_size):
        obs, extras, options, rewards, next_obs, next_extras, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.from_numpy(np.stack(obs)),
            torch.from_numpy(np.stack(extras)),
            torch.from_numpy(np.stack(options)),
            torch.from_numpy(np.stack(rewards)).squeeze(-1),
            torch.from_numpy(np.stack(next_obs)),
            torch.from_numpy(np.stack(next_extras)),
            torch.from_numpy(np.stack(dones, dtype=np.long)),
        )

    def __len__(self):
        return len(self.buffer)
