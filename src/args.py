import argparse


parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument("--env", default="CartPole-v0", help="ROM to run")
parser.add_argument("--optimal-eps", type=float, default=0.05, help="Epsilon when playing optimally")
parser.add_argument("--frame-skip", default=4, type=int, help="Every how many frames to process")
parser.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount rate")
parser.add_argument("--epsilon-start", type=float, default=1.0, help=("Starting value for epsilon."))
parser.add_argument("--epsilon-min", type=float, default=0.1, help="Minimum epsilon.")
parser.add_argument("--epsilon-decay", type=float, default=20000, help=("Number of steps to minimum epsilon."))
parser.add_argument("--max-history", type=int, default=10000, help=("Maximum number of steps stored in replay"))
parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
parser.add_argument("--freeze-interval", type=int, default=200, help=("Interval between target freezes."))
parser.add_argument("--update-frequency", type=int, default=4, help=("Number of actions before each SGD update."))
parser.add_argument("--termination-reg", type=float, default=0.01, help=("Regularization to decrease termination prob."))
parser.add_argument("--entropy-reg", type=float, default=0.01, help=("Regularization to increase policy entropy."))
parser.add_argument("--num-options", type=int, default=2, help=("Number of options to create."))
parser.add_argument("--temp", type=float, default=1, help="Action distribution softmax tempurature param.")

parser.add_argument("--max_steps_ep", type=int, default=18000, help="number of maximum steps per episode.")
parser.add_argument("--max_steps_total", type=int, default=int(4e6), help="number of maximum steps to take.")  # bout 4 million
parser.add_argument("--cuda", type=bool, default=True, help="Enable CUDA training (recommended if possible).")
parser.add_argument("--seed", type=int, default=0, help="Random seed for numpy, torch, random.")
parser.add_argument("--logdir", type=str, default="runs", help="Directory for logging statistics")
parser.add_argument("--exp", type=str, default=None, help="optional experiment name")
parser.add_argument("--switch-goal", type=bool, default=False, help="switch goal after 2k eps")
