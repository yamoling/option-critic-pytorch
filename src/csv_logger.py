import os
import time
import csv
import numpy as np
import polars as pl
from typing import Any, Optional

QVALUES = "qvalues.csv"
TRAIN = "train.csv"
TEST = "test.csv"
TRAINING_DATA = "training_data.csv"
ACTIONS = "actions.json"
PID = "pid"

# Dataframe columns
TIME_STEP_COL = "time_step"
TIMESTAMP_COL = "timestamp_sec"


class CSVWriter:
    def __init__(self, filename: str, flush_interval_sec: float = 30):
        self.filename = filename
        self._file = None
        self._writer = None
        self._flush_interval = flush_interval_sec
        self._next_flush = time.time() + flush_interval_sec
        self._schema = {
            TIME_STEP_COL: "int",
            TIMESTAMP_COL: "float",
        }

    def log(self, data: dict[str, Any], time_step: int):
        if len(data) == 0:
            return
        now = time.time()
        data["timestamp_sec"] = now
        data["time_step"] = time_step
        if self._writer is None:
            if os.path.exists(self.filename):
                self._file = open(self.filename, "a")
            else:
                self._file = open(self.filename, "w")
            self._writer = csv.DictWriter(self._file, fieldnames=data.keys())
            self._writer.writeheader()
        try:
            self._writer.writerow(data)
        except ValueError:
            # Occurs when the header has changed compared to the previous data
            self._reformat(data)
            self._writer.writerow(data)
        if now >= self._next_flush:
            assert self._file is not None
            self._file.flush()
            self._next_flush = now + self._flush_interval

    def _reformat(self, data: dict[str, float]):
        """
        When trying to write a data item whose columns do not match the header, we need to
        re-write the while. We choose to pad the columns with None values.

        Note: this is costly if reformatting happens when the file is already large.
        """
        self.close()
        df = pl.read_csv(self.filename)
        new_headers = set(data.keys()) - set(df.columns)
        df = df.with_columns([pl.lit(None).alias(h) for h in new_headers])
        df.write_csv(self.filename)

        self._file = open(self.filename, "a")
        self._writer = csv.DictWriter(self._file, fieldnames=df.columns)

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()


class CSVLogger:
    def __init__(self, logdir: str, flush_interval_sec: float = 30):
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        self.data = CSVWriter(os.path.join(logdir, TRAINING_DATA), flush_interval_sec)
        self.train = CSVWriter(os.path.join(logdir, TRAIN), flush_interval_sec)

    def log_train(self, data: dict[str, float], time_step: int):
        return self.train.log(data, time_step)

    def log_training_data(self, data: dict[str, float], time_step: int):
        return self.data.log(data, time_step)

    def log_episode(self, step_num: int, score: float, option_lengths: dict, ep_num: int, epsilon: float, exit_rate: float):
        log_entry = {
            "step_num": step_num,
            "score": score,
            "ep_num": ep_num,
            "epsilon": epsilon,
            "exit_rate": exit_rate,
        }
        for option, lens in option_lengths.items():
            # Need better statistics for this one, point average is terrible in this case
            if len(lens) > 0:
                log_entry[f"option_{option}_avg_length"] = float(np.mean(lens))
                log_entry[f"option_{option}_active"] = sum(lens) / ep_num
            else:
                log_entry[f"option_{option}_avg_length"] = 0.0
                log_entry[f"option_{option}_active"] = 0.0
        self.log_train(log_entry, step_num)

    def log_data(self, step: int, actor_loss: Optional[float], critic_loss: Optional[float], entropy: float, epsilon: float):
        metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "policy_entropy": entropy,
            "epsilon": epsilon,
        }
        self.log_training_data(metrics, step)
