import time
import random
from copy import deepcopy

import polars as pl
import numpy as np
import torch

from args import parser
from experience_replay import ReplayBuffer
from logger import Logger
from option_critic import OptionCriticFeatures
from option_critic import actor_loss as actor_loss_fn
from option_critic import critic_loss as critic_loss_fn
from utils import to_tensor
import lle
from typing import Any
from marlenv import RLEnvWrapper, Builder


class FRL(RLEnvWrapper[Any]):
    def __init__(self):

        env1 = lle.from_file("fourrooms.toml").obs_type("state").build()
        env2 = lle.from_file("fourrooms2.toml").obs_type("state").build()
        self.world1 = env1.world
        self.world2 = env2.world
        self.env1 = Builder(env1).time_limit(1000).agent_id().build()
        self.env2 = Builder(env2).time_limit(1000).agent_id().build()
        super().__init__(self.env1)

    def switch_goal(self):
        if self.wrapped is self.env1:
            self.wrapped = self.env2
            self.world = self.world2
        else:
            self.wrapped = self.env1
            self.world = self.world1

    @property
    def current_pos(self):
        return self.world.agents_positions[0]


def run(args):
    env = FRL()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    option_critic = OptionCriticFeatures(
        in_features=env.observation_size + env.extras_size,
        num_actions=env.n_actions,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device,
        method=args.method,
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)
    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    buffer = ReplayBuffer(capacity=args.max_history)
    logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}")

    steps = 0
    epsilon = 1.0
    ep_lengths = []
    step_logs, ep_logs = [], []
    while steps < args.max_steps_total:
        cumulative_reward = 0.0
        option_lengths = {opt: [] for opt in range(args.num_options)}

        obs, _ = env.reset()
        obs = obs.agent(0, keep_dim=False)
        available = obs.available_actions
        obs = np.concatenate([obs.data, obs.extras], axis=-1)
        state = option_critic.get_state(to_tensor(obs))
        greedy_option = option_critic.greedy_option(state)
        current_option = 0

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).
        if args.switch_goal and logger.n_eps == 1000:
            env.switch_goal()

        done = False
        ep_steps = 0
        option_termination = True
        curr_op_len = 0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0

            action, logp, entropy = option_critic.get_action(state, current_option, available[None, :])

            step = env.step([action])
            next_obs = step.obs.agent(0, keep_dim=False)
            next_obs = np.concatenate([next_obs.data, next_obs.extras], axis=-1)
            reward = step.reward.item()
            done = step.done
            buffer.push(obs, current_option, reward, next_obs, done)
            cumulative_reward += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(
                    obs, current_option, logp, entropy, reward, done, next_obs, option_critic, option_critic_prime, args
                )
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()
                step_logs.append(
                    {
                        "actor loss": actor_loss.item() if actor_loss else None,
                        "critic loss": critic_loss.item() if critic_loss else None,
                        "entropy": entropy.item(),
                        "epsilon": epsilon,
                        "time step": steps,
                    }
                )

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs
            available = step.obs.available_actions[0]

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        ep_lengths.append(ep_steps)
        mean_len = np.mean(ep_lengths[-50:])
        ep_logs.append(
            {
                "score": cumulative_reward,
                "length": ep_steps,
                "mean length": mean_len,
                **{f"option {i} length": np.mean(option_lengths[i]) if option_lengths[i] else 0 for i in range(args.num_options)},
            }
        )
        logger.log_episode(steps, cumulative_reward, ep_steps, mean_len, epsilon)
    return step_logs, ep_logs


if __name__ == "__main__":
    args = parser.parse_args()
    args.max_steps_total = 10_000
    for method in ("params", "module list"):
        args.method = method
        for seed in range(10):
            args.seed = seed
            step_logs, ep_logs = run(args)
            step_df = pl.DataFrame(step_logs).with_columns(method=pl.lit(method), seed=pl.lit(seed))
            ep_df = pl.DataFrame(ep_logs).with_columns(method=pl.lit(method), seed=pl.lit(seed))
            step_df.write_csv(f"results/step_logs_{method}_{seed}.csv")
            ep_df.write_csv(f"results/ep_logs_{method}_{seed}.csv")
