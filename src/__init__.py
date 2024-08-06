# import Imported_Code.admm_joint_pruning
import torch
from . import cfg
from . import modelstruct
from . import datareader
from . import filemanagement


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
    recordModelInfo(model, logger)

    return {"model": model, "logger": logger, "data": data, "config": config}


def swapping_run(config: cfg.ConfigObject | bool | None = None, **kwargs):
    # Get the defaults
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    model: modelstruct.SwappingDetectionModel = modelstruct.getModel(config) if "model" not in kwargs else kwargs["model"]
    if not isinstance(model, modelstruct.SwappingDetectionModel):
        return
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
    recordModelInfo(model, logger)
    # addm_test(config, model=model, logger=logger, data=data)

    return {"model": model, "logger": logger, "data": data, "config": config}


def addm_test(config: cfg.ConfigObject | bool | None = None, **kwargs):

    from . import Imported_Code
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

    # Reset the optimizer to include the v layers (This is to emulate the original code, not sure if it is in the paper)
    optimizer = model.cfg("Optimizer")(model.parameters(), lr=model.cfg("LearningRate"))

    # Adds an interpretation layer to the config so that it can be read
    wrapped_cfg = Imported_Code.ConfigCompatabilityWrapper(config)

    # Performs the pruning method
    Imported_Code.prune_admm(wrapped_cfg, model, config("Device"), data, data, optimizer)

    # Applies the pruning to the base model, NOTE: MIGHT CAUSE ISSUES WITH "Imported_Code.remove_addm_v_layers"
    # Imported_Code.apply_filter(model, config("Device"), wrapped_cfg)  # Commenting out because I think this is redundent
    mask = Imported_Code.apply_prune(model, config("Device"), wrapped_cfg)

    # Removes the added v layers from the model
    Imported_Code.remove_addm_v_layers(model)

    # print(mask)
    # Calculates the weights that should be kept stable because they are pruned
    frozen = {name: torch.ones_like(m, requires_grad=False) for name, m in mask.items()}
    frozen = {name: weight*frozen[name] for name, weight in model.named_parameters() if name in frozen.keys()}
    model.frozen = model.frozen | frozen

    config("PruningSelection", "ADDM_Joint")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
    model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))
    print(model.fit(config("NumberOfEpochs")))
    recordModelInfo(model, logger)

    return {"model": model, "logger": logger, "data": data, "config": config}


def thinet_test_old(config: cfg.ConfigObject | bool | None = None, **kwargs):
    from . import Imported_Code

    # Get the defaults
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    model: modelstruct.SwappingDetectionModel = modelstruct.getModel(config) if "model" not in kwargs else kwargs["model"]
    # logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]
    # data = datareader.get_dataset(config) if "data" not in kwargs else kwargs["data"]

    for i in range(len(config("WeightPrunePercent")) - 1):
        Imported_Code.thinet_pruning(model, i, config=config)

    config("PruningSelection", "thinet")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
    model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))
    print(model.fit(config("NumberOfEpochs")))
    recordModelInfo(model, logger)

    return {"model": model, "logger": logger, "config": config}


def thinet_test(config: cfg.ConfigObject | bool | None = None, **kwargs):
    from . import Imported_Code

    # Get the defaults
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    model: modelstruct.SwappingDetectionModel = modelstruct.getModel(config) if "model" not in kwargs else kwargs["model"]
    # logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]
    data = datareader.get_dataset(config) if "data" not in kwargs else kwargs["data"]

    # Create sample of dataset, Fancy load_kwargs is just there to load the collate_fn
    training_data = iter(torch.utils.data.DataLoader(data.dataset, 100000, **(data.dataset.load_kwargs if hasattr(data.dataset, "load_kwargs") else {}))).__next__()[0]

    for i in range(len(config("WeightPrunePercent")) - 1):
        module_results: list[Imported_Code.forward_hook] = Imported_Code.collect_module_is(model, [i, i+1], training_data)
        x = module_results[0].inp.detach()
        y = module_results[0].out_no_bias.detach()

        if (x.ndim > 2 and x.shape[1] != 1):
            x = torch.sum(x, dim=-1)
        elif x.shape[1] == 1:
            x = torch.flatten(x, start_dim=1)

        if (y.ndim > 2 and y.shape[1] != 1):
            y = torch.sum(y, dim=-1)
        elif y.shape[1] == 1:
            y = torch.flatten(y, start_dim=1)

        indexes, weight_mod = Imported_Code.value_sum(x, y, config("WeightPrunePercent")[i-1])

        keep_tensor = torch.zeros_like(x[0], dtype=torch.bool)
        keep_tensor[indexes] = True

        Imported_Code.remove_layers(model, i, keepint_tensor=keep_tensor)

    config("PruningSelection", "thinet")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
    model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))
    print(model.fit(config("NumberOfEpochs")))
    recordModelInfo(model, logger)

    return {"model": model, "logger": logger, "config": config}


def recordModelInfo(model: modelstruct.BaseDetectionModel, logger: filemanagement.ExperimentLineManager):
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())
    logger("NumberOfZeros", model.get_zero_weights())
    logger("ModelWeightStructure", model.get_model_structure(count_zeros=True))
    logger("ModelWeightStructurePruneZero", model.get_model_structure())


if __name__ == "__main__":
    # This no longer works.
    args = standard_run()
    # swapping_run()
    thinet_test(**args)
    addm_test(**args)
