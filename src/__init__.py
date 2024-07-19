import cfg
import modelstruct
import datareader
import filemanagement


def standard_run(config: cfg.ConfigObject | bool | None = None):
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    model = modelstruct.BaseDetectionModel()
    logger = filemanagement.ExperimentLineManager(cfg=config)
    data = datareader.get_dataset(config)
    model.set_training_data(data)
    model.cfg = config

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    print(model.fit(10))
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())


if __name__ == "__main__":
    config = cfg.ConfigObject()
    model = modelstruct.BaseDetectionModel()
    logger = filemanagement.ExperimentLineManager(cfg=config)
    data = datareader.get_dataset(config)
    model.set_training_data(data)
    model.cfg = config
    # model.epoch_callbacks.append(lambda x: (print(x)))

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))
    # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
    model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))

    print(model.fit(10))
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())
