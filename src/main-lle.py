import numpy as np
import torch
import marlenv
from lle import LLE
from shaped_doors import ShapedDoors
from copy import deepcopy
from random import random
from csv_logger import CSVLogger
from option_critic import OptionCriticConvLLE
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
import multiprocessing as mp
from experience_replay import ReplayBuffer
from logger import Logger

from args import Args


def run(args: Args):
    # env = LLE.from_file("doors").obs_type("layered").state_type("state").build()
    # env = marlenv.Builder(env).time_limit(env.width * env.height // 2).build()

    env = ShapedDoors(args.reward_delay)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"New run with seed={args.seed} and delay={args.reward_delay}")
    assert len(env.observation_shape) == 3
    option_critic = OptionCriticConvLLE(
        input_shape=env.observation_shape,
        extras_size=env.extras_shape[0],
        num_actions=env.n_actions,
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = CSVLogger(logdir=f"logs/delay-{args.reward_delay}-seed-{args.seed}")

    step_num = 0
    while step_num < args.max_steps_total:
        score = 0
        option_lengths = {opt: [] for opt in range(args.num_options)}

        observation, _ = env.reset()
        next_obs = torch.from_numpy(observation.data).to(device)
        next_extras = torch.from_numpy(observation.extras).to(device)
        next_available_actions = torch.from_numpy(observation.available_actions).to(device)
        greedy_option = 0
        current_option = 0

        done = False
        ep_length = 0
        option_termination = True
        curr_op_len = 0
        while not done:
            obs = next_obs
            extras = next_extras
            available_actions = next_available_actions
            epsilon = option_critic.epsilon(step_num)

            state = option_critic.get_state(obs, extras)
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)
            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                if random() < epsilon:
                    current_option = np.random.choice(args.num_options)
                else:
                    current_option = greedy_option
                curr_op_len = 0

            action, logp, entropy = option_critic.get_action(
                state=state,
                available_actions=available_actions,
                option=current_option,
            )
            step = env.step(action)
            done = step.done
            reward = step.reward.item()
            next_obs = torch.from_numpy(step.obs.data).to(device)
            next_available_actions = torch.from_numpy(step.obs.available_actions).to(device)
            next_extras = torch.from_numpy(step.obs.extras).to(device)
            buffer.push(obs, extras, current_option, reward, next_obs, next_extras, step.done)
            score += reward

            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(
                    obs,
                    extras,
                    current_option,
                    logp,
                    entropy,
                    reward,
                    step.done,
                    next_obs,
                    next_extras,
                    option_critic,
                    option_critic_prime,
                    args,
                )
                loss = actor_loss
                critic_loss = None
                if step_num % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss
                    critic_loss = critic_loss.item()  # for logging

                optim.zero_grad()
                loss.backward()
                optim.step()

                if step_num % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())
                logger.log_data(step_num, actor_loss.item(), critic_loss, entropy.item(), epsilon)
            # update global steps etc
            step_num += 1
            ep_length += 1
            curr_op_len += 1

        exit_rate = step.info["exit_rate"]
        logger.log_episode(step_num, score, option_lengths, ep_length, epsilon, exit_rate)


if __name__ == "__main__":
    all_args = list[Args]()
    for seed in range(10):
        for delay in [0, 1, 2, 3, 4, 100]:
            all_args.append(Args(seed=seed, reward_delay=delay))

    with mp.Pool(processes=5) as pool:
        pool.map(run, all_args)
