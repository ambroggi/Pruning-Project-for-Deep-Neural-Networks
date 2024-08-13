# import Imported_Code.admm_joint_pruning
import torch
from . import cfg
from . import modelstruct
from . import datareader
from . import filemanagement
from . import Imported_Code


def standard_run(config: cfg.ConfigObject | bool | None = None, **kwargs):
    # Get the defaults
    if config is None:
        config = cfg.ConfigObject.get_param_from_args()
    elif not config:
        config = cfg.ConfigObject()
    else:
        config = config.clone()
    model = modelstruct.getModel(config) if "model" not in kwargs else kwargs["model"]
    logger = filemanagement.ExperimentLineManager(cfg=config) if "logger" not in kwargs else kwargs["logger"]
    data = datareader.get_dataset(config) if "data" not in kwargs else kwargs["data"]

    # Model set up
    if "modelStateDict" in kwargs.keys():
        model.load_model_state_dict_with_structure(kwargs["modelStateDict"])
    model.set_training_data(data)
    model.cfg = config

    # Save all parts
    kwargs = kwargs | {"model": model, "logger": logger, "data": data, "config": config}

    if "PruningSelection" in kwargs.keys() and kwargs["PruningSelection"] is not None:
        kwargs = types_of_tests[kwargs["PruningSelection"]](**kwargs)
        model = kwargs["model"]
        logger = kwargs["logger"]
        data = kwargs["data"]
        config = kwargs["config"]

        # Making sure this is the same as well
        model.cfg = config

    # Sometimes want to run for a while without logging (retraining runs)
    if logger is not None:
        # This is just adding things to the log
        model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

        # This is complicated. It is adding only the mean loss to the log, but only when the epoch number is even, and it is adding it as a new row.
        model.epoch_callbacks.append(lambda x: ([logger(f"Epoch {x['epoch']} {a}", b) for a, b in x.items() if a in ["mean_loss"]] if (x["epoch"] % 2 == 0) else None))

    print(model.fit(config("NumberOfEpochs") if "NumberOfEpochs" not in kwargs else kwargs["NumberOfEpochs"]))

    if logger is not None:
        recordModelInfo(model, logger)

    return kwargs | {"model": model, "logger": logger, "data": data, "config": config}


def swapping_run(config: cfg.ConfigObject, model: torch.nn.Module, data, layers: list[int] | None = None, **kwargs):
    if layers is None:
        layers = range(len(config("WeightPrunePercent"))-1)

    targets = []
    currents = []
    path_for_layer = []

    for i in layers:
        # This is the swapping
        state_dict = model.state_dict_of_layer_i(i)
        changing_weight_and_bias = list(state_dict.keys())
        path_for_layer.append(changing_weight_and_bias[0])
        targets.append(int(len(state_dict[changing_weight_and_bias[0]])*config("WeightPrunePercent")[i]))
        currents.append(len(state_dict[changing_weight_and_bias[0]]))

    # This is so inefficent
    while True in [a > b for a, b in zip(currents, targets)]:
        for i in layers:
            if currents[i] > targets[i]:
                state_dict = model.state_dict_of_layer_i(i+1)

                currents[i] = max(targets[i], currents[i] - config("LayerPruneTargets")[i])

                old_layer = Imported_Code.get_layer_by_state(model, path_for_layer[i])
                if isinstance(old_layer, torch.nn.Linear):
                    Imported_Code.set_layer_by_state(model, path_for_layer[i], torch.nn.Linear(old_layer.in_features, currents[i]))
                elif isinstance(old_layer, torch.nn.Conv1d):
                    Imported_Code.set_layer_by_state(model, path_for_layer[i], torch.nn.Conv1d(old_layer.in_channels, currents[i], old_layer.kernel_size))

                for name in state_dict:
                    if isinstance(state_dict[name], torch.Tensor):
                        state_value: torch.Tensor = state_dict[name]
                        if len(state_value.shape) == 2:
                            if isinstance(old_layer, torch.nn.Conv1d):
                                # This is just to check if going from CNN layer to FC and scale
                                state_value = state_value[:, :(len(state_value[0])//old_layer.out_channels) * currents[i]]
                            else:
                                state_value = state_value[:, :currents[i]]
                        elif len(state_value.shape) == 3:
                            state_value = state_value[:, :currents[i], :]

                        state_dict[name] = state_value

                model.load_model_state_dict_with_structure(state_dict)

                # Run a standard run
                kw = kwargs | {"model": model, "logger": None, "PruningSelection": None, "data": data, "config": config}
                kw.pop("modelStateDict")  # keep changes from being overrided
                standard_run(**kw)

    config("PruningSelection", "iteritive_full_theseus")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return {"model": model, "logger": logger, "data": data, "config": config}


def addm_test(model: torch.nn.Module, config: cfg.ConfigObject, data, **kwargs):
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

    return kwargs | {"model": model, "logger": logger, "data": data, "config": config}


def thinet_test_old(config: cfg.ConfigObject, model: torch.nn.Module, layers: list[int] | None = None, **kwargs):
    if layers is None:
        layers = range(len(config("WeightPrunePercent")) - 1)

    for i in layers:
        Imported_Code.thinet_pruning(model, i, config=config)

    config("PruningSelection", "thinet_recreation")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "config": config, "layers": layers}


def thinet_test(config: cfg.ConfigObject, data: torch.utils.data.DataLoader, model: torch.nn.Module, layers: list[int] | None = None, **kwargs):
    if layers is None:
        layers = range(len(config("WeightPrunePercent")) - 1)

    # Create sample of dataset, Fancy load_kwargs is just there to load the collate_fn
    training_data = iter(torch.utils.data.DataLoader(data.dataset, 100000, **(data.dataset.load_kwargs if hasattr(data.dataset, "load_kwargs") else {}))).__next__()[0]

    for i in layers:
        Imported_Code.run_thinet_on_layer(model, i, training_data=training_data, config=config)

    config("PruningSelection", "thinet")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "config": config, "data": data, "layers": layers}


def bert_of_theseus_test(model: modelstruct.BaseDetectionModel, data, config: cfg.ConfigObject, **kwargs):
    lst = [model.conv1, model.pool1, model.conv2]

    start, end = Imported_Code.forward_hook(), Imported_Code.forward_hook()
    rm1 = lst[0].register_forward_hook(start)
    rm2 = lst[-1].register_forward_hook(end)

    # Create sample of dataset, Fancy load_kwargs is just there to load the collate_fn
    training_data = iter(torch.utils.data.DataLoader(data.dataset, 100, **(data.dataset.load_kwargs if hasattr(data.dataset, "load_kwargs") else {}))).__next__()[0]

    model(training_data)

    rm1.remove()
    rm2.remove()

    replace_object = Imported_Code.Theseus_Replacement(lst, start.inp.shape[1:], end.out.shape[1:], model=model)

    model.fit(10)

    replace_object.condense_in_model(model)

    config("PruningSelection", "BERT_theseus")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "data": data, "config": config, "logger": logger}


def DAIS_test(model: modelstruct.BaseDetectionModel, data, config: cfg.ConfigObject, layers: list[int] | None = None, **kwargs):
    # Find the layers to apply it to
    if layers is None:
        layers = range(1, len(config("WeightPrunePercent")))

    alphas = []
    layer = -1
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            layer += 1

            if layer in layers:
                alphas.append(Imported_Code.add_alpha(module, 0.2, config("NumberOfEpochs")))

    model.epoch_callbacks.extend([a.callback_fn for a in alphas])

    config("PruningSelection", "DAIS_training")
    logger = filemanagement.ExperimentLineManager(cfg=config)
    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    Imported_Code.DAIS_fit(model, alphas, epochs=10)

    for a in alphas:
        a.remove()

    config("PruningSelection", "DAIS")

    logger = filemanagement.ExperimentLineManager(cfg=config)

    model.train(False)

    return kwargs | {"model": model, "data": data, "config": config, "logger": logger}


def recordModelInfo(model: modelstruct.BaseDetectionModel, logger: filemanagement.ExperimentLineManager):
    logger("macs", model.get_FLOPS())
    logger("parameters", model.get_parameter_count())
    logger("NumberOfZeros", model.get_zero_weights())
    logger("ModelWeightStructure", model.get_model_structure(count_zeros=True))
    logger("ModelWeightStructurePruneZero", model.get_model_structure())


types_of_tests = {
    "ADDM_Joint": addm_test,
    "thinet_recreation": thinet_test_old,
    "thinet": thinet_test,
    "iteritive_full_theseus": swapping_run,
    "BERT_theseus": bert_of_theseus_test,
    "DAIS": DAIS_test
}


if __name__ == "__main__":
    # This no longer works.
    args = standard_run()
    # swapping_run()
    thinet_test(**args)
    addm_test(**args)
