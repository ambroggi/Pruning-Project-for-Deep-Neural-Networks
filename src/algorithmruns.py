# This file is supposed to be all of the different algorithms that can be run

import torch

import math
from . import Imported_Code
from . import cfg
from . import modelstruct
from . import getdata
from . import filemanagement
from . import extramodules


def swapping_run(config: cfg.ConfigObject, model: modelstruct.BaseDetectionModel, data, layers: list[int] | None = None, **kwargs):
    if layers is None:
        layers = [layernum for layernum, layerpercent in enumerate(config("WeightPrunePercent")) if layerpercent < 1]

    targets = []
    currents = []
    path_for_layer = []

    for i in layers:
        # This is for finding the target dimentions
        state_dict = model.state_dict_of_layer_i(i)  # get the state dict of current layer
        weights_path = [x for x in state_dict.keys() if "weight" in x][0]  # find what the weights are called (it is in the form "x.weight")
        path_for_layer.append(weights_path)  # save the path to weights so we dont need to calculate it again
        targets.append(math.ceil(len(state_dict[weights_path])*config("WeightPrunePercent")[i]))  # find how small it should be pruned to
        currents.append(len(state_dict[weights_path]))  # save the shape it currently is

    config("PruningSelection", "iteritive_full_theseus_training")

    # This is so inefficent, it loops until the layer sizes are at least as small as the targets
    while True in [a > b for a, b in zip(currents, targets)]:
        for i in layers:
            if currents[i] > targets[i]:
                # Reduce layer i

                # First get layer i+1's state dictionary
                state_dict = model.state_dict_of_layer_i(i+1)

                # check the planned reduction and save it
                currents[i] = max(targets[i], currents[i] - config("LayerIteration")[i])

                # Get the actual Module from the model, I made a function to get it from the state dictionary key
                old_layer = Imported_Code.get_layer_by_state(model, path_for_layer[i])

                # These need to be unique because I don't know of any generic cloning for the layers
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

                # fit the new model
                model.fit(epochs=model.cfg("NumberOfEpochs"))

    config("PruningSelection", "iteritive_full_theseus")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "data": data, "config": config}


def addm_test(model: modelstruct.BaseDetectionModel, config: cfg.ConfigObject, data, **kwargs):
    # Adds the compatability to the model that is needed
    Imported_Code.add_addm_v_layers(model)

    # Reset the optimizer to include the v layers (This is to emulate the original code, not sure if it is in the paper)
    optimizer = model.cfg("Optimizer")(model.parameters(), lr=model.cfg("LearningRate"))

    # Adds an interpretation layer to the config so that it can be read
    wrapped_cfg = Imported_Code.ConfigCompatabilityWrapper(config, translations="ADDM")

    # Expects model to have log_softmax applied (observed from the model used in the code)
    remover = model.register_forward_hook(lambda module, args, output: torch.nn.functional.log_softmax(output, dim=1))

    # Performs the pruning method
    Imported_Code.prune_admm(wrapped_cfg, model, config("Device"), data, data, optimizer)

    # Applies the pruning to the base model, NOTE: MIGHT CAUSE ISSUES WITH "Imported_Code.remove_addm_v_layers"
    # Imported_Code.apply_filter(model, config("Device"), wrapped_cfg)  # Commenting out because I think this is redundent
    mask = Imported_Code.apply_prune(model, config("Device"), wrapped_cfg)

    # Set all weights to their new values
    with torch.no_grad():
        # Only keeps the top magnitude weights as described in section "4.4. Network retraining"
        for percent_to_prune, w in zip(config("WeightPrunePercent"), model.get_important_modules()):
            w: torch.nn.Linear
            indices = torch.topk(abs(w.weight.flatten()), int(len(w.weight.flatten())*(1-percent_to_prune)), largest=False).indices
            w.weight.view(-1)[indices] = 0

        # Removes the added v layers from the model
        Imported_Code.remove_addm_v_layers(model)

    # Remove F.log_softmax
    remover.remove()

    # print(mask)
    # Calculates the weights that should be kept stable because they are pruned
    frozen = {name: torch.ones_like(m, requires_grad=False) for name, m in mask.items()}
    frozen = {name: weight*frozen[name] for name, weight in model.named_parameters() if name in frozen.keys()}
    model.frozen = model.frozen | frozen

    config("PruningSelection", "ADDM_Joint")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "data": data, "config": config}


def thinet_test_old(config: cfg.ConfigObject, model: modelstruct.BaseDetectionModel, layers: list[int] | None = None, **kwargs):
    if layers is None:
        layers = [layernum for layernum, layerpercent in enumerate(config("WeightPrunePercent")) if layerpercent < 1]

    for i in layers:
        Imported_Code.thinet_pruning(model, i, config=config)

    config("PruningSelection", "thinet_recreation")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "config": config, "layers": layers}


def thinet_test(config: cfg.ConfigObject, data: torch.utils.data.DataLoader, model: modelstruct.BaseDetectionModel, layers: list[int] | None = None, **kwargs):
    if layers is None:
        layers = [layernum for layernum, layerpercent in enumerate(config("WeightPrunePercent")) if layerpercent < 1]

    # Create sample of dataset, Fancy load_kwargs is just there to load the collate_fn
    training_data = iter(torch.utils.data.DataLoader(data.dataset, 100000, **(data.dataset.load_kwargs if hasattr(data.dataset, "load_kwargs") else {}))).__next__()[0]

    for i in layers:
        Imported_Code.run_thinet_on_layer(model, i, training_data=training_data, config=config)

    config("PruningSelection", "thinet")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "config": config, "data": data, "layers": layers}


def bert_of_theseus_test(model: modelstruct.BaseDetectionModel, data, config: cfg.ConfigObject, **kwargs):
    start = model.get_important_modules()[0]
    end = model.get_important_modules()[-1]
    lst: list[torch.nn.Module] = [m for m in model.modules()]
    lst = lst[lst.index(start): lst.index(end)]
    # lst = [model.conv1, model.pool1, model.conv2, model.flatten, model.fc1]

    start, end = Imported_Code.forward_hook(), Imported_Code.forward_hook()
    rm1 = lst[0].register_forward_hook(start)
    rm2 = lst[-1].register_forward_hook(end)

    # Create sample of dataset, Fancy load_kwargs is just there to load the collate_fn
    training_data = iter(torch.utils.data.DataLoader(data.dataset, 100, **(data.dataset.load_kwargs if hasattr(data.dataset, "load_kwargs") else {}))).__next__()[0]

    model(training_data)

    rm1.remove()
    rm2.remove()

    replace_object = Imported_Code.Theseus_Replacement(lst, start.inp.shape[1:], end.out.shape[1:], model=model)

    config("PruningSelection", "BERT_theseus_training")
    model.fit(10)

    replace_object.condense_in_model(model)

    config("PruningSelection", "BERT_theseus")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "data": data, "config": config, "logger": logger}


def DAIS_test(model: modelstruct.BaseDetectionModel, data, config: cfg.ConfigObject, layers: list[int] | None = None, **kwargs):
    # Find the layers to apply it to
    if layers is None:
        layers = [layernum for layernum, layerpercent in enumerate(config("WeightPrunePercent")) if layerpercent < 1]

    alphas = []
    for layer, module in enumerate(model.get_important_modules()):
        if layer in layers:
            alphas.append(Imported_Code.add_alpha(module, config("WeightPrunePercent")[layer], config("NumberOfEpochs")))

    model.epoch_callbacks.extend([a.callback_fn for a in alphas])

    config("PruningSelection", "DAIS_training")
    logger = filemanagement.ExperimentLineManager(cfg=config, pth="results/extra.csv")
    # This is just adding thigns to the log
    model.epoch_callbacks.append(lambda x: ([logger(a, b, can_overwrite=True) for a, b in x.items()]))

    Imported_Code.DAIS_fit(model, alphas, epochs=10)

    for a in alphas:
        a.remove()

    config("PruningSelection", "DAIS")

    logger = filemanagement.ExperimentLineManager(cfg=config)

    model.train(False)

    return kwargs | {"model": model, "data": data, "config": config, "logger": logger}


def TOFD_test(model: modelstruct.BaseDetectionModel, data, config: cfg.ConfigObject, layers: list[int] | None = None, **kwargs):
    wrap = Imported_Code.task_oriented_feature_wrapper(model)

    optimizer = config("Optimizer")(wrap.parameters(), lr=config("LearningRate"))
    train1, train2 = getdata.get_train_test(config, dataloader=data)
    args = Imported_Code.ConfigCompatabilityWrapper(config=config, translations="TOFD")
    new_net = modelstruct.getModel(config)

    # Creating new state dict
    new_net_state_dict = {}
    prior_percentage = 1
    for module_num, module in enumerate(model.get_important_modules()):
        state_dict = new_net.state_dict_of_layer_i(module_num)
        percentage = config("WeightPrunePercent")[module_num]
        current_length = len(module.bias)
        new_net_state_dict.update({x: y[:math.ceil(len(y)*percentage), :math.ceil(len(y[0])*prior_percentage)] for x, y in state_dict.items() if ("weight" in x)})
        new_net_state_dict.update({x: y[:math.ceil(len(y)*percentage)] for x, y in state_dict.items() if ("bias" in x)})
        prior_percentage = math.ceil(current_length*percentage)/current_length

    # print(*[f"{x}:{y.shape}\n" for x, y in model.state_dict().items()])
    # print(*[f"{x}:{y.shape}\n" for x, y in new_net_state_dict.items()])
    new_net.load_model_state_dict_with_structure(new_net_state_dict)

    # Model set up
    train, validation = getdata.get_train_test(config, dataloader=data)
    new_net.set_training_data(train)
    new_net.set_validation_data(validation)
    new_net.cfg = config

    new_wrap = Imported_Code.task_oriented_feature_wrapper(new_net)

    Imported_Code.TOFD_name_main(optimizer=optimizer, teacher=wrap, net=new_wrap, trainloader=train1, testloader=train2, device=config("Device"), args=args, epochs=config("NumberOfEpochs"), LR=config("LearningRate"), criterion=config("LossFunction")())

    wrap.remove()

    config("PruningSelection", "TOFD_INCOMPLETE")  # Incomplete because the auxillary modules are not exactly as described

    logger = filemanagement.ExperimentLineManager(cfg=config)

    new_net.train(False)

    return kwargs | {"model": new_net, "data": data, "config": config, "logger": logger}


def Random_test(model: modelstruct.BaseDetectionModel, config: cfg.ConfigObject, **kwargs):
    for count, module in enumerate(model.get_important_modules()):
        pruning_layer = (extramodules.PostMutablePruningLayer(module))
        n = len(pruning_layer.para)
        random_permutation = torch.randperm(n)
        random_filter: torch.Tensor = random_permutation.less((config("WeightPrunePercent")[count]*n)//1)
        pruning_layer.para.data = random_filter.type_as(pruning_layer.para.data)
        pruning_layer.remove(update_weights=True)

    config("PruningSelection", "RandomStructured")

    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "config": config, "logger": logger}


types_of_tests = {
    "ADDM_Joint": addm_test,
    "thinet_recreation": thinet_test_old,
    "thinet": thinet_test,
    "iteritive_full_theseus": swapping_run,
    "BERT_theseus": bert_of_theseus_test,
    "DAIS": DAIS_test,
    "TOFD": TOFD_test,
    "RandomStructured": Random_test
}
