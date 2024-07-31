# import Imported_Code.admm_joint_pruning
import torch
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
    model = modelstruct.getModel(config) if "model" not in kwargs else kwargs["model"]
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
    model: modelstruct.SwappingDetectionModel = modelstruct.getModel(config) if "model" not in kwargs else kwargs["model"]
    logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]
    data = datareader.get_dataset(config) if "data" not in kwargs else kwargs["data"]

    # Run a standard run
    standard_run(config, model=model, logger=logger, data=data)

    # This is the swapping
    model.swap_testlhidden(50)

    config("PruningSelection", "Theseus")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
    model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))
    print(model.fit(config("NumberOfEpochs")))
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())
    # addm_test(config, model=model, logger=logger, data=data)
    thinet_test(config, model=model, logger=logger, data=data)


def addm_test(config: cfg.ConfigObject | bool | None = None, **kwargs):

    import Imported_Code
    # Get the defaults
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    model: modelstruct.SwappingDetectionModel = modelstruct.getModel(config) if "model" not in kwargs else kwargs["model"]
    # logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]
    data = datareader.get_dataset(config) if "data" not in kwargs else kwargs["data"]

    # Adds the compatability to the model that is needed
    Imported_Code.add_addm_v_layers(model)
    # Adds an interpretation layer to the config so that it can be read
    wrapped_cfg = Imported_Code.ConfigCompatabilityWrapper(config)

    # Performs the pruning method
    Imported_Code.prune_admm(wrapped_cfg, model, config("Device"), data, data, model.optimizer)

    # Applies the pruning to the base model, NOTE: MIGHT CAUSE ISSUES WITH "Imported_Code.remove_addm_v_layers"
    Imported_Code.apply_filter(model, config("Device"), wrapped_cfg)
    mask = Imported_Code.apply_l1_prune(model, config("Device"), wrapped_cfg)

    # Removes the added v layers from the model
    Imported_Code.remove_addm_v_layers(model)

    print(mask)
    # Calculates the weights that should be kept stable because they are pruned
    frozen = {name: torch.ones_like(m, requires_grad=False)-m for name, m in mask.items()}
    frozen = {name: weight*frozen[name] for name, weight in model.named_parameters() if name in frozen.keys()}
    model.frozen = model.frozen | frozen

    config("PruningSelection", "ADDM_Joint")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
    model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))
    print(model.fit(config("NumberOfEpochs")))
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())


def thinet_test(config: cfg.ConfigObject | bool | None = None, **kwargs):
    import Imported_Code

    # Get the defaults
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    model: modelstruct.SwappingDetectionModel = modelstruct.getModel(config) if "model" not in kwargs else kwargs["model"]
    # logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]
    # data = datareader.get_dataset(config) if "data" not in kwargs else kwargs["data"]

    Imported_Code.thinet_pruning(model, 2, config=config)

    config("PruningSelection", "thinet")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
    model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))
    print(model.fit(config("NumberOfEpochs")))
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())



if __name__ == "__main__":
    # standard_run()
    swapping_run()


