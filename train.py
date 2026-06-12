"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *
from lavis.common.annotator.uniformer.mmcv.utils.logging import get_logger, logger_initialized, print_log

os.environ["HTTP_PROXY"]  = "http://localhost:911"
os.environ["HTTPS_PROXY"] = "http://localhost:911"
os.environ["http_proxy"]  = "http://localhost:911"
os.environ["https_proxy"] = "http://localhost:911"
os.environ["NO_PROXY"]    = "localhost,127.0.0.1,10.0.0.0/8,.renault.fr"
os.environ["no_proxy"]    = "localhost,127.0.0.1,10.0.0.0/8,.renault.fr"

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def init_distributed_mode(run_cfg):

    # Get the initialized logger, if not exist,
    # create a logger named `mmcv`
    logger_names = list(logger_initialized.keys())
    logger_name = logger_names[0] if logger_names else 'mmcv'

    if "LOCAL_RANK" in os.environ:
        run_cfg.gpu = int(os.environ["LOCAL_RANK"])
        run_cfg.rank = int(os.environ["RANK"])
        run_cfg.world_size = int(os.environ["WORLD_SIZE"])
    else:
        print_log('Not using distributed mode', logger=logger_name)
        run_cfg.distributed = False
        run_cfg.gpu = 0
        return

    run_cfg.distributed = True
    torch.cuda.set_device(run_cfg.gpu)
    dist_url = "env://"
    print_log(f'| distributed init (rank {run_cfg.rank}): {dist_url}', logger=logger_name)
    torch.distributed.init_process_group(
        backend="nccl", 
        init_method=dist_url,
        world_size=run_cfg.world_size,
        rank=run_cfg.rank,
        timeout=datetime.timedelta(days=365),
    )
    torch.distributed.barrier()


def cleanup_distributed_mode(run_cfg):
    if not getattr(run_cfg, "distributed", False):
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # Get the initialized logger, if not exist,
    # create a logger named `mmcv`
    logger_names = list(logger_initialized.keys())
    logger_name = logger_names[0] if logger_names else 'mmcv'

    cfg = Config(parse_args())
    print_log("Config loaded", logger=logger_name)

    init_distributed_mode(cfg.run_cfg)
    try:
        setup_seeds(cfg)

        # set after init_distributed_mode() to only log on master.
        setup_logger()

        cfg.pretty_print()

        task = tasks.setup_task(cfg)
        print_log("Task setup completed", logger=logger_name)
        datasets = task.build_datasets(cfg)
        print_log("Datasets built", logger=logger_name)
        model = task.build_model(cfg)
        print_log("Model built", logger=logger_name)

        print_log(msg = ("nb of parameters: %d", sum(p.numel() for p in model.parameters())), logger=logger_name)
        print_log(msg = ("nb of trainable parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad)), logger=logger_name)

        runner = get_runner_class(cfg)(
            cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
        )
        runner.train()
    finally:
        cleanup_distributed_mode(cfg.run_cfg)


if __name__ == "__main__":
    main()
