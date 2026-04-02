import time
import random
from copy import deepcopy

import numpy as np
import torch

from args import parser
from experience_replay import ReplayBuffer
from fourrooms import Fourrooms
from logger import Logger
from option_critic import OptionCriticFeatures
from option_critic import actor_loss as actor_loss_fn
from option_critic import critic_loss as critic_loss_fn
from utils import to_tensor


def run(args):
    env = Fourrooms()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    option_critic = OptionCriticFeatures(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device,
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)
    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    buffer = ReplayBuffer(capacity=args.max_history)
    logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}")

    steps = 0
    epsilon = 1.0
    if args.switch_goal:
        print(f"Current goal {env.goal}")
    while steps < args.max_steps_total:
        rewards = 0
        option_lengths = {opt: [] for opt in range(args.num_options)}

        obs = env.reset()
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

            action, logp, entropy = option_critic.get_action(state, current_option)

            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

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

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)


if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
