# import Imported_Code.admm_joint_pruning
import torch
import gc
import time
import psutil

from . import cfg
from . import modelstruct
from . import getdata
from . import filemanagement
from .algorithmruns import types_of_tests


# This may be useful: https://stackoverflow.com/a/53593326


def standard_run(config: cfg.ConfigObject | bool | None = None, save_epoch_waypoints: bool = False, from_savepoint: None | int = None, **kwargs) -> dict[str, any]:
    # Get the defaults
    # Priority: 1) savepoint, 2) config being None (None meaning not set yet), 3) config being False (false being do not set config), 4) normal config
    if isinstance(from_savepoint, int):
        savepoint = standardLoad(from_savepoint)
        config = savepoint.pop("config")
        kwargs = kwargs | savepoint
    elif config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    else:
        config = config.clone()
    data = getdata.get_dataloader(config) if "data" not in kwargs else kwargs["data"]
    model = modelstruct.getModel(config) if "model" not in kwargs else kwargs["model"]

    # Model set up
    if "modelStateDict" in kwargs.keys():
        model.load_model_state_dict_with_structure(kwargs["modelStateDict"])
    train, validation = getdata.get_train_test(config, dataloader=data)
    model.set_training_data(train)
    model.set_validation_data(validation)
    model.cfg = config

    # Save all parts
    kwargs = kwargs | {"model": model, "data": data, "config": config}

    t = time.time()
    mem = psutil.virtual_memory()[3]/1000000
    cuda_mem = torch.cuda.memory_allocated()
    if "PruningSelection" in kwargs.keys() and kwargs["PruningSelection"] is not None:
        model.train(True)
        model.to(model.cfg("Device"))
        kwargs = types_of_tests[kwargs["PruningSelection"]](**kwargs)
        kwargs.pop("PruningSelection")
        model = kwargs["model"]
        logger = kwargs["logger"]
        data = kwargs["data"]
        config = kwargs["config"]

        # Making sure this is the same as well
        model.cfg = config
    else:
        logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]

    epochs: int = config("NumberOfEpochs") if "NumberOfEpochs" not in kwargs else kwargs["NumberOfEpochs"]

    if model.training and epochs > 0:
        # Make a series of 5 (or fewer) evenly spaced points along the epochs for waypoints
        epoch_waypoints = (list(range(epochs)[-2::-epochs//5]) if epochs > 2 else [])[::-1]
        # This is complicated. It is adding only the mean loss to the log, but only for each epoch waypoint, and it is adding it as a new row.
        model.epoch_callbacks.append(lambda results: ([logger(f"Epoch waypoint {epoch_waypoints.index(results['epoch'])} {name_of_value}", value) for name_of_value, value in results.items() if name_of_value in ["mean_loss"]] if (results["epoch"] in epoch_waypoints) else None))
        if save_epoch_waypoints:
            # This is saving the model, but only during epoch waypoints.
            model.epoch_callbacks.append(lambda results: model.save_model_state_dict(logger, update_config=False, logger_column=f"-Waypoint_{epoch_waypoints.index(results['epoch'])}-") if results['epoch'] in epoch_waypoints else None)

    model.fit(epochs)

    # Sometimes want to run for a while without logging (retraining runs)
    if logger is not None:
        logger("TimeForPrune", time.time()-t)
        logger("Memory", psutil.virtual_memory()[3]/1000000-mem)
        logger("CudaMemory", torch.cuda.memory_allocated()-cuda_mem)
        logger("LengthOfTrainingData", len(train.dataset))
        logger("LengthOfValidationData", len(validation.dataset))
        if "prior_logger_row" in kwargs:
            logger("AssociatedOriginalRow", kwargs["prior_logger_row"])

        # This is just adding things to the log
        model.epoch_callbacks.append(lambda results: ([logger(name_of_value, value, can_overwrite=True) for name_of_value, value in results.items()]))

    t = time.time()
    print(model.fit())

    if logger is not None:
        logger("TimeForRun", time.time()-t)
        recordModelInfo(model, logger)

    # Create the model state
    # The 'hasattr(b, "clone")' is for strings
    # The 'if "total" not in a' is because the FLOPS count apparently saves itself as a Parameter for some reason
    model_state = {"modelStateDict": {a: (b.clone() if hasattr(b, "clone") else b) for a, b in model.state_dict().items() if "total" not in a}}
    logger_row = {"prior_logger_row": logger.row_id} if logger is not None else {}
    return kwargs | {"model": model, "logger": logger, "data": data, "config": config} | logger_row | model_state


def recordModelInfo(model: modelstruct.BaseDetectionModel, logger: filemanagement.ExperimentLineManager):
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())
    logger("NumberOfZeros", model.get_zero_weights())
    filter_in_out = model.get_zero_filters()
    logger("Zeroed_filter_inputs", filter_in_out[0])
    logger("Zeroed_filter_outputs", filter_in_out[1])
    logger("ModelWeightStructure", model.get_model_structure(count_zeros=True))
    logger("ModelWeightStructurePruneZero", model.get_model_structure())

    # for name, x in model.named_parameters():
    #     if ("total_ops" in name) or ("total_params" in name):
    #         del x

    garbage_sum = 0
    # found code for checking garbage collection (removed because other objects were causing errors, see v0.63): https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            garbage_sum += sum(obj.size())

    logger("GarbageCollectionSizeTotal", garbage_sum)


def standardLoad(index: None | int = None, existing_config: cfg.ConfigObject | None = None) -> dict[str, any]:
    config, index = filemanagement.load_cfg(config=existing_config) if index is None else filemanagement.load_cfg(row_number=index, config=existing_config)
    if config("SaveLocation") is not None:
        modelStateDict = torch.load("savedModels/"+config("SaveLocation"), map_location=config("Device"))
        config("FromSaveLocation", config("SaveLocation"))
        config("SaveLocation", "None")
        return {"config": config, "modelStateDict": modelStateDict, "prior_logger_row": index}
    return {"config": config, "prior_logger_row": index}
