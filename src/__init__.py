import cfg
import modelstruct
import datareader
import filemanagement


def standard_run(config: cfg.ConfigObject | bool | None = None, **kwargs):
    # Get the defaults
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    model = modelstruct.getModel(config("ModelStructure")) if "model" not in kwargs else kwargs["model"]
    logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]
    data = datareader.get_dataset(config) if "data" not in kwargs else kwargs["data"]

    # Model set up
    model.set_training_data(data)
    model.cfg = config

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
    model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))

    print(model.fit(config("NumberOfEpochs")))
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())

    return {"model": model, "logger": logger, "data": data}


def swapping_run(config: cfg.ConfigObject | bool | None = None, **kwargs):
    # Get the defaults
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    model: modelstruct.SwappingDetectionModel = modelstruct.getModel(config("ModelStructure")) if "model" not in kwargs else kwargs["model"]
    logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]
    data = datareader.get_dataset(config) if "data" not in kwargs else kwargs["data"]

    # Run a standard run
    standard_run(config, model=model, logger=logger, data=data)

    # This is the swapping
    model.swap_testlhidden(50)
    logger = filemanagement.ExperimentLineManager(cfg=config)

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
    model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))
    print(model.fit(config("NumberOfEpochs")))
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())
    addm_test(config, model=model, logger=logger, data=data)


def addm_test(config: cfg.ConfigObject | bool | None = None, **kwargs):
    # from sys import path
    # print(path)
    import Imported_Code
    # Get the defaults
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    model: modelstruct.SwappingDetectionModel = modelstruct.getModel(config("ModelStructure")) if "model" not in kwargs else kwargs["model"]
    # logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]
    data = datareader.get_dataset(config) if "data" not in kwargs else kwargs["data"]

    Imported_Code.prune_admm(Imported_Code.ConfigCompatabilityWrapper(config), model, config("Device"), data, data, model.optimizer)


if __name__ == "__main__":
    # standard_run()
    swapping_run()
