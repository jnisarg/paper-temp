import yaml
import shutil
import random
import argparse
import numpy as np
from pathlib import Path

from collections import deque

import torch
import torch.backends.cudnn as cudnn


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config file path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int):
    assert seed >= 0, "Seed must be non-negative"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True


def initialize_logger(
    work_dir: str,
    exp_name: str,
    logger_name: str = "train",
    use_tb: bool = False,
    tb_name: str = "tb_logs",
):
    work_dir = Path(work_dir).joinpath(exp_name)

    if work_dir.exists() and logger_name == "train":
        if (
            input(
                f"{work_dir} already exists, it will be overwritten... Are you sure? [y/n]: "
            )
            != "y"
        ):
            exit(0)

        shutil.rmtree(str(work_dir))
        work_dir.mkdir(parents=True)

    from loguru import logger
    from rich.logging import RichHandler

    logger.remove()
    logger.configure(
        handlers=[
            {
                "sink": RichHandler(),
                "format": "<level>{message}</level>",
            }
        ]
    )
    logger.add(
        f"{work_dir}/{logger_name}.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        level="INFO",
    )

    writer = None
    if use_tb:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(work_dir) + f"/{tb_name}")

    return str(work_dir), logger, writer


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt if fmt else "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.nanmedian().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.nanmean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
        )
