import logging
import os
import time
from typing import Optional
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logdir, run_name):
        self.log_name = os.path.join(logdir, run_name)
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + "/logger.log"),
            ],
            datefmt="%Y/%m/%d %I:%M:%S %p",
        )

    def log_episode(self, step_num: int, score: float, option_lengths: dict, ep_num: int, epsilon: float, exit_rate: float):
        self.n_eps += 1
        logging.info(
            f"> ep {self.n_eps} done. total_steps={step_num} | score={score} | episode_steps={ep_num} "
            f"| hours={(time.time() - self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f}"
        )
        self.writer.add_scalar(tag="episodic_rewards", scalar_value=score, global_step=step_num)
        self.writer.add_scalar(tag="episode_lengths", scalar_value=ep_num, global_step=step_num)
        self.writer.add_scalar(tag="exit_rate", scalar_value=exit_rate, global_step=step_num)
        # Keep track of options statistics
        for option, lens in option_lengths.items():
            # Need better statistics for this one, point average is terrible in this case
            if len(lens) > 0:
                mean = np.mean(lens)
                ssum = sum(lens) / ep_num
            else:
                mean = 0.0
                ssum = 0.0
            self.writer.add_scalar(tag=f"option_{option}_avg_length", scalar_value=mean, global_step=step_num)
            self.writer.add_scalar(tag=f"option_{option}_active", scalar_value=ssum, global_step=step_num)

    def log_data(self, step: int, actor_loss: Optional[float], critic_loss: Optional[float], entropy: float, epsilon: float):
        if actor_loss:
            self.writer.add_scalar(tag="actor_loss", scalar_value=actor_loss, global_step=step)
        if critic_loss:
            self.writer.add_scalar(tag="critic_loss", scalar_value=critic_loss, global_step=step)
        self.writer.add_scalar(tag="policy_entropy", scalar_value=entropy, global_step=step)
        self.writer.add_scalar(tag="epsilon", scalar_value=epsilon, global_step=step)
