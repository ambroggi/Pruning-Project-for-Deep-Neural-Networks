# used the tutorial for this: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

# https://www.reddit.com/r/learnpython/comments/vupzfa/comment/iffaj95/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os

# from functools import partial
from ray import train, tune
# from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler

from src import cfg, getdata, modelstruct

# from src import filemanagement


# import ray.cloudpickle as pickle


def search_space_config(config: cfg.ConfigObject):
    modifications = {
        "LearningRate": tune.loguniform(1e-4, 1e-1),
        "BatchSize": tune.choice(10**x for x in range(1, 8)),
        "Dropout": tune.loguniform(1e-4, 1e-1),
        "HiddenDim": tune.randint(0, 100),
        "HiddenDimSize": tune.randint(10, 1000),
        "MaxSamples": 0,
        "NumberOfEpochs": 100
    }
    return config.to_dict() | modifications


def raytrain(config: cfg.ConfigObject | bool | None = None, **kwargs) -> dict[str, any]:
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    else:
        config = config.clone()

    conf = search_space_config(config) | {"DatafolderPath": os.path.join(os.getcwd(), "datasets")}

    scheduler = ASHAScheduler(
        metric="val_f1_score_macro",
        mode="max",
        max_t=config("NumberOfEpochs"),
        grace_period=3,
        reduction_factor=2,
    )

    result = tune.run(
        singleraytrain,
        resources_per_trial={"cpu": 2, "gpu": 0.36 if "data" not in kwargs else kwargs["gpus"]},
        config=conf,
        num_samples=25,
        scheduler=scheduler,
        max_concurrent_trials=5,
        storage_path=os.path.abspath("results/raytraining"),
        log_to_file="logRaytrace.txt",
        max_failures=3,
        time_budget_s=82800)

    best = result.get_best_trial("total_loss", "min", "last")
    print(best)
    print(best.config)

    best = result.get_best_trial("val_f1_score_macro", "max", "last")
    print(best)
    print(best.config)


def singleraytrain(conf_dict, **kwargs):
    modelstruct.torch.autograd.set_detect_anomaly(True)
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
