import os
import sys
import math

import torch

from torchdrug import core, tasks
from torchdrug.utils import comm, pretty

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reasoning import dataset, layer, model, task, util


def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return

    if hasattr(cfg.train, "batch_per_epoch"):
        step = 1
    else:
        step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1
    dataset_class = cfg.dataset["class"]
    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        metric = solver.evaluate("valid")
        # solver.evaluate("test")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
            solver.save(f"/mnt/newdisk/linyawei/AStarNet/checkpoint/practice.pth")

    return solver


def test(cfg, solver):
    # dataset_class = cfg.dataset["class"]
    # checkpoint = torch.load(f"/home/nudt/linyawei/AStarNet/checkpoint/{dataset_class}.pth")
    # print("Checkpoint keys:", checkpoint['model'].keys())
    # print("Model state_dict keys:", solver.model.state_dict().keys())

    # for key in checkpoint['model']:
    #     print(f"{key}: {checkpoint['model'][key].shape}")
    # for key in solver.model.state_dict():
    #     print(f"{key}: {solver.model.state_dict()[key].shape}")
    solver.load(f"/mnt/newdisk/linyawei/AStarNet/checkpoint/practice.pth")
    solver.evaluate("valid")
    solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pretty.format(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    print(cfg.dataset)
    train_and_validate(cfg, solver)
    test(cfg, solver)
    