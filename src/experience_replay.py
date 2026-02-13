import random
from collections import deque
import torch


class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, extras, option, reward, next_obs, next_extras, done):
        self.buffer.append((obs, extras, option, reward, next_obs, next_extras, done))

    def sample(self, batch_size):
        obs, extras, option, reward, next_obs, next_extras, done = zip(*self.rng.sample(self.buffer, batch_size))
        return torch.cat(obs), torch.cat(extras), option, reward, torch.cat(next_obs), torch.cat(next_extras), done

    def __len__(self):
        return len(self.buffer)
