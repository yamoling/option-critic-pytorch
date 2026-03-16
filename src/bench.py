import torch
import timeit

import torch.nn as nn


# Create a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


INPUT_SIZE = 784
HIDDEN_SIZE = 256


def no_grad(linear, x):
    with torch.no_grad():
        linear.forward(x)


def with_grad(linear, x):
    linear.forward(x)


def main():
    for eeval in (True, False):
        linear = torch.nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 10),
        )
        linear.train(eeval)
        # if eeval:
        #     linear.eval()
        for batch_size in (16, 32, 64, 128):
            x = torch.randn(batch_size, INPUT_SIZE, device=torch.device("cpu"))
            duration_nograd = timeit.timeit(lambda: no_grad(linear, x), number=5000)
            duration_grad = timeit.timeit(lambda: with_grad(linear, x), number=5000)

            print(f"Batch size={batch_size}:")
            print(f"\tTime without torch.no_grad(): {duration_grad:.4f}s")
            print(f"\tTime with torch.no_grad(): {duration_nograd:.4f}s")
            print(f"\tSpeedup: {duration_grad / duration_nograd:.2f}x")


if __name__ == "__main__":
    main()
