# import Imported_Code.admm_joint_pruning
import gc
import time

import psutil
import torch

from . import cfg, filemanagement, getdata, modelstruct
from .algorithmruns import types_of_tests

# This may be useful: https://stackoverflow.com/a/53593326


def standard_run(config: cfg.ConfigObject | bool | None = None, save_epoch_waypoints: bool = False, data: getdata.torch.utils.data.DataLoader | None = None, model: modelstruct.BaseDetectionModel | None = None, PruningSelection: str | None = None, from_savepoint: None | int = None, **kwargs) -> dict[str, any]:
    """This is the main running function that is used to both train the models and run different algorithms. It calls the specific algorithms using PruningSelection before training (or retraining if using an algorithm).
    It does assume that any model you set PruningSelection for is already trained.

    Args:
        config (cfg.ConfigObject | bool | None, optional): The config object to attach to the model. Defaults to None.
        save_epoch_waypoints (bool, optional): A booliean to check if you want to save the model at 5 intermidate points during training. Defaults to False.
        data (getdata.torch.utils.data.DataLoader | None, optional): This is a dataloader to use with the model, automatically splits it into train/test. (generates one from config if none is given) Defaults to None.
        model (modelstruct.BaseDetectionModel | None, optional): This is the model you want to use assuming you don't want to use a config default one. Defaults to None.
        PruningSelection (str | None, optional): Algorithm to use, called pruning selection because that is what it was called in the config. Defaults to None.
        from_savepoint (None | int, optional): Where to load from using standard load. Overrides data and model. Defaults to None.
        modelStateDict (None | dict[str, torch.Tensor]: (hidden) The model state dictionary to load into the active model, mostly used internally for laoding from standardLoad). Defaults to None.
        NumberOfEpochs (None | int): (hidden) overrides the number of epochs for the model. Defaults to None.
        logger (None | filemanagement.ExperimentLineManager): (hidden) Internal currently active logging line. Usually creates a new line, but the last line can be kept if you want. None means to use no logger at all and leave the run unlogged. Defaults to Not in kwargs.
        prior_logger_row (int): (hidden) The last row written to the log file. Used to create links through standardLoad so the graphing can know what model each row was used with. Defaults to Not in kwargs.

    Returns:
        dict[str, any]: The kwargs needed to rerun this run. Besides PruningSelection being removed, so that you can run a new pruning method. At a minimum has:
            model: Same as input
            logger: Logger used to record the most recent run, defaults to None or generated logger, so using the kwargs without modifications will result in no new logger being created.
            data: Same as input
            config: Same as input
            modelStateDict: Same as input
            prior_logger_row: Same as input, but is generated if it did not exist and logger was not None.
    """
    # Get the defaults
    # Priority: 1) savepoint, 2) config being None (None meaning not set yet), 3) config being False (false being do not set config), 4) normal config
    if isinstance(from_savepoint, int):
        savepoint = standardLoad(from_savepoint)
        config = savepoint.pop("config")
        kwargs = kwargs | savepoint
        # Make sure to use the new versions of args (by regenerating them from the new config)
        data = None
        model = None
    elif config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    else:
        config = config.clone()

    if data is None:
        data = getdata.get_dataloader(config)

    if model is None:
        model = modelstruct.getModel(config)

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
    if PruningSelection is not None:
        model.train(True)
        model.to(model.cfg("Device"))
        kwargs = types_of_tests[PruningSelection](**kwargs)
        assert "PruningSelection" not in kwargs  # Just a test to make sure I didn't accidentally add this anywhere
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

    if logger is not None:
        logger("TimeForPrune", time.time()-t)

    model.fit(epochs)

    # Sometimes want to run for a while without logging (retraining runs)
    if logger is not None:
        logger("TimeForPruneAndRetrain", time.time()-t)
        logger("Memory", psutil.virtual_memory()[3]/1000000-mem)
        logger("CudaMemory", torch.cuda.memory_allocated()-cuda_mem)
        logger("LengthOfTrainingData", len(train.dataset))
        logger("LengthOfValidationData", len(validation.dataset))
        if "prior_logger_row" in kwargs:
            logger("AssociatedOriginalRow", kwargs["prior_logger_row"])

        # This is just adding things to the log
        model.epoch_callbacks.append(lambda results: ([logger(name_of_value, value, can_overwrite=True) for name_of_value, value in results.items()]))

    model.train(False)
    model.eval()
    t = time.time()
    print(model.fit(epochs))

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
    """
    Records the model information in the log, such as the parameter counts and the zeroed out filters.

    Args:
        model (modelstruct.BaseDetectionModel): Model to identify the model parameters of.
        logger (filemanagement.ExperimentLineManager): Logger linking to a log file to write to.
    """
    logger("macs", model.get_macs())
    logger("parameters", model.get_parameter_count())
    logger("NumberOfZeros", model.get_zero_weights())
    filter_in_out = model.get_zero_filters()
    logger("Zeroed_filter_inputs", filter_in_out[0])
    logger("Zeroed_filter_outputs", filter_in_out[1])
    logger("ModelWeightStructure", model.get_model_structure(count_zeros=True))
    logger("ModelWeightStructurePruneZero", model.get_model_structure())

    garbage_sum = 0
    # found code for checking garbage collection (removed because other objects were causing errors, see v0.63): https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            garbage_sum += sum(obj.size())

    logger("GarbageCollectionSizeTotal", garbage_sum)


def standardLoad(index: None | int = None, existing_config: cfg.ConfigObject | None = None) -> dict[str, any]:
    """
    This is the standard method that we have for loading a previously run model. It can load either a specific index from the results file, or a specific pytorch tensor file that contains the model weights.
    It returns them in a dictionary of keywords that the standard run can read and make use of.

    Args:
        index (None | int, optional): Index of the config to load, this is ignored if config(FromSaveLocation) is not None. Defaults to None.
        existing_config (cfg.ConfigObject | None, optional): This is the existing config you want to use that has FromSaveLocation read from it and is passed back if FromSaveLocation is a file location. Defaults to None.

    Returns:
        dict[str, any]: A dictionary of keyword arguments for the standard run function. Can contain "config", "prior_logger_row", "modelStateDict", and/or "loadedfrom".
            "config" is a config object that should match whatever row was read from the results csv
            "prior_logger_row" is the index the config was loaded from so that connections can be made between the different logging rows.
            "modelStateDict" is the weights and biases from the loaded model set to be loaded when the model next starts running.
            "loadedfrom" is the file name that the model was loaded from (only appears if loaded from a specific file)
    """

    # Set up the kwargs for the load_cfg function because passing None does not get the default value
    load_kwargs = {"config": existing_config}
    if index is not None:
        load_kwargs["row_number"] = index

    if existing_config is not None:  # Just a guard to check that existing_config(value) won't cause an error
        # Check if you want to load a specific thing, such as a specific row of csv or a specific pytorch file
        if existing_config("FromSaveLocation") is not None:
            if existing_config("FromSaveLocation").split(" ")[0] == "csv":
                # This is if you are loading from a command line defined csv row, replaces index
                load_kwargs["row_number"] = int(existing_config("FromSaveLocation").split(" ")[1])
                existing_config("FromSaveLocation", "None")
            else:
                print("Specific file loaded, standardLoad has been skipped")
                existing_config("Notes", existing_config("Notes") | 16)
                return {"config": existing_config, "loadedfrom": existing_config("FromSaveLocation")}

        if existing_config("ResultsPath") is not None:
            load_kwargs["pth"] = existing_config("ResultsPath")

    config, index = filemanagement.load_cfg(**load_kwargs)

    if config("SaveLocation") is not None:  # prior row actually has a model to load.
        modelStateDict = torch.load("savedModels/"+config("SaveLocation"), map_location=config("Device"))
        config("FromSaveLocation", config("SaveLocation"))
        config("SaveLocation", "None")
        return {"config": config, "modelStateDict": modelStateDict, "prior_logger_row": index}
    return {"config": config, "prior_logger_row": index}
