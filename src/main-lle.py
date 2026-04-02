import numpy as np
import torch
from torch import nn
import random
from csv_logger import CSVLogger
from option_critic import OptionCriticTrainer
import multiprocessing as mp
from lle import LLE
from datetime import datetime
from args import Args
from marlenv import Transition
from networks import CNNOptionCritic, CNNStateEncoder
from tqdm import tqdm


def run(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = LLE.from_file("four_rooms.toml").obs_type("layered").builder().agent_id().time_limit(1_000).build()
    # env = LLE.from_file("doors").obs_type("layered").state_type("state").build()
    # env = marlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2).buisld()
    env.seed(args.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"New run with seed={args.seed} and delay={args.reward_delay}")
    assert len(env.observation_shape) == 3
    HIDDEN_SIZE = 512
    option_critic = CNNOptionCritic(
        args.num_options,
        CNNStateEncoder(env, device),
        nn.ModuleList([nn.Linear(HIDDEN_SIZE, env.n_actions) for _ in range(args.num_options)]),
        nn.Sequential(nn.Linear(HIDDEN_SIZE, args.num_options), nn.Sigmoid()),
        nn.Linear(HIDDEN_SIZE, args.num_options),  # Policy-Over-Options
    )

    trainer = OptionCriticTrainer(
        option_critic,
        device,
        args.max_history,
        args.learning_rate,
        args.gamma,
        args.entropy_reg,
        args.termination_reg,
        args.update_frequency,
        args.batch_size,
        args.freeze_interval,
        env.n_agents,
    )
    logger = CSVLogger(logdir=f"logs/{datetime.now().isoformat()}")

    agent = trainer.make_agent()
    step_num = 0
    episode = 0
    pbar = tqdm(total=args.max_steps_total, desc="Training", unit="step")
    exit_rates = []
    while step_num < args.max_steps_total:
        episode += 1
        observation, state = env.reset()
        done = False
        exit_rate = 0.0
        ep_length = 0
        while not done:
            ep_length += 1
            action, logp, entropy = agent.choose_action(observation)
            step = env.step(action)
            done = step.done
            trainer.update_step(
                Transition.from_step(
                    observation,
                    state,
                    action,
                    step,
                    options=agent.current_options,
                    logp=logp,
                    entropy=entropy,
                ),
                step_num,
            )
            step_num += 1
            exit_rate = step.info.get("exit_rate", 0.0)
            observation = step.obs
            pbar.update()
        exit_rates.append(exit_rate)
        print(f"Episode {episode} finished with exit rate {exit_rate:.2f} in {ep_length} steps.")
        pbar.set_postfix({"mean exit rate": np.mean(exit_rates[-50:]), "episode": episode})
        logger.log_episode(step_num, 0, 0, agent.epsilon, exit_rate)


def main():
    all_args = list[Args]()
    n_processes = 1
    for seed in range(1):
        for delay in [0]:
            args = Args(seed=seed, reward_delay=delay, num_options=5, max_steps_total=200_000)
            all_args.append(args)
    if n_processes > 1:
        with mp.Pool(processes=5) as pool:
            return pool.map(run, all_args)
    return [run(args) for args in all_args]


if __name__ == "__main__":
    main()
