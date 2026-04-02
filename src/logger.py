import logging
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logdir, run_name):
        self.log_name = logdir + "/" + run_name
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + "/logger.log"),
            ],
            datefmt="%Y/%m/%d %I:%M:%S %p",
        )

    def log_episode(self, steps, reward, ep_steps, mean_len, epsilon):
        self.n_eps += 1
        logging.info(
            f"> ep {self.n_eps}, step={steps}\t| reward={reward}\t| len={ep_steps}\t| mean len={mean_len:.2f}\t| epsilon={epsilon:.3f}"
        )

    def log_data(self, step, actor_loss, critic_loss, entropy, epsilon):
        if actor_loss:
            self.writer.add_scalar(tag="actor_loss", scalar_value=actor_loss.item(), global_step=step)
        if critic_loss:
            self.writer.add_scalar(tag="critic_loss", scalar_value=critic_loss.item(), global_step=step)
        self.writer.add_scalar(tag="policy_entropy", scalar_value=entropy, global_step=step)
        self.writer.add_scalar(tag="epsilon", scalar_value=epsilon, global_step=step)
