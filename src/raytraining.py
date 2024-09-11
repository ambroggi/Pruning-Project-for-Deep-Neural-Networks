# used the tutorial for this: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

# https://www.reddit.com/r/learnpython/comments/vupzfa/comment/iffaj95/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src import cfg
from src import modelstruct
from src import getdata
# from src import filemanagement

import os

# from functools import partial
from ray import tune
from ray import train
# from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
# import ray.cloudpickle as pickle


def searchspaceconfig(config: cfg.ConfigObject):
    modifications = {
        "LearningRate": tune.loguniform(1e-4, 1e-1),
        "BatchSize": tune.choice(10**x for x in range(8)),
        "Dropout": tune.loguniform(1e-4, 1e-1),
        "HiddenDim": tune.randint(0, 100),
        "HiddenDimSize": tune.randint(10, 1000),
        "MaxSamples": 0
    }
    return config.to_dict() | modifications


def raytrain(config: cfg.ConfigObject | bool | None = None, **kwargs) -> dict[str, any]:
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    else:
        config = config.clone()

    conf = searchspaceconfig(config) | {"DatafolderPath": os.path.join(os.getcwd(), "datasets")}

    scheduler = ASHAScheduler(
        metric="total_loss",
        mode="min",
        max_t=config("NumberOfEpochs"),
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        singleraytrain,
        resources_per_trial={"cpu": 2, "gpu": 0.1 if "data" not in kwargs else kwargs["gpus"]},
        config=conf,
        num_samples=9,
        scheduler=scheduler,
    )

    best = result.get_best_trial("total_loss", "min", "last")
    print(best)


def singleraytrain(conf_dict, **kwargs):
    cwdpath = conf_dict.pop("DatafolderPath")
    config = cfg.ConfigObject.from_dict(conf_dict)

    # Model set up
    # This is where I would load checkpoints if I wanted to.
    getdata.datasets_folder_path = cwdpath
    data = getdata.get_dataloader(config=config)
    training, testing = getdata.get_train_test(config, dataloader=data)
    train_data, validation = getdata.get_train_test(config, dataloader=training)

    model = modelstruct.getModel(config)
    model.set_training_data(train_data)
    model.set_validation_data(validation)
    model.cfg = config

    model.epoch_callbacks.append(train.report)

    model.fit(config("NumberOfEpochs") if "NumberOfEpochs" not in kwargs else kwargs["NumberOfEpochs"])


if __name__ == "__main__":
    raytrain()
