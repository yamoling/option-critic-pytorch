from dataclasses import dataclass


@dataclass
class Args:
    optimal_eps: float = 0.05
    """The optimal epsilon value for exploration in epsilon-greedy policy."""
    learning_rate: float = 0.0005
    """Learning rate for the optimizer."""
    gamma: float = 0.99
    """Discount factor for future rewards."""
    epsilon_start: float = 1.0
    """Initial value of epsilon for exploration."""
    epsilon_min: float = 0.05
    """Minimum value of epsilon after decay."""
    epsilon_decay: int = 50_000
    """Number of steps over which epsilon decays."""
    max_history: int = 50_000
    """Maximum size of the replay buffer."""
    batch_size: int = 32
    """Number of samples per training batch."""
    freeze_interval: int = 200
    """Interval (in steps) to update the target network."""
    update_frequency: int = 5
    """Frequency (in steps) to update the network."""
    termination_reg: float = 0.01
    """Regularization coefficient for option termination."""
    entropy_reg: float = 0.01
    """Regularization coefficient for entropy."""
    num_options: int = 2
    """Number of options (policies) available."""
    temp: float = 1
    """Temperature parameter for softmax."""
    max_steps_total: int = 200_000
    """Maximum total steps for training."""
    cuda: bool = True
    """Whether to use CUDA for computation."""
    seed: int = 0
    """Random seed for reproducibility."""
    logdir: str = "runs"
    """Directory to save logs and outputs."""
    reward_delay: int = 0
    """Delay (in steps) for reward signals."""
