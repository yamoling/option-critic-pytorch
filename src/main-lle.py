import numpy as np
import torch
from torch import nn
import random
from csv_logger import CSVLogger
from option_critic import OptionCriticTrainer
import multiprocessing as mp
from lle import LLE
from args import Args
from marlenv import Transition
from networks import CNNOptionCritic, CNNStateEncoder


def run(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = LLE.from_file("four_rooms.toml").obs_type("layered").builder().agent_id().time_limit(78).build()
    # env = LLE.from_file("doors").obs_type("layered").state_type("state").build()
    # env = marlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2).buisld()
    env.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
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
    logger = CSVLogger(logdir=f"logs/delay-{args.reward_delay}-seed-{args.seed}")

    agent = trainer.make_agent()
    step_num = 0
    while step_num < args.max_steps_total:
        score = 0.0
        observation, state = env.reset()
        done = False
        exit_rate = 0.0
        while not done:
            action, logp, entropy = agent.choose_action(observation)
            step = env.step(action)
            done = step.done
            score += step.reward.item()
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
        logger.log_episode(step_num, score, 0, agent.epsilon, exit_rate)


def main():
    all_args = list[Args]()
    n_processes = 1
    for seed in range(10):
        for delay in [0, 1, 2, 3, 4, 100]:
            args = Args(seed=seed, reward_delay=delay, num_options=5)
            all_args.append(args)
    if n_processes > 1:
        with mp.Pool(processes=5) as pool:
            return pool.map(run, all_args)
    return [run(args) for args in all_args]


if __name__ == "__main__":
    main()
