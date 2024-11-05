# This file is supposed to be all of the different algorithms that can be run

import math

import torch
import torch.utils
import torch.utils.hooks
from tqdm import tqdm

from . import (Imported_Code, cfg, extramodules, filemanagement,
               modelstruct)


def swapping_run(config: cfg.ConfigObject, model: modelstruct.BaseDetectionModel, layers: list[int] | None = None, **kwargs):
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

    tq = tqdm(total=sum(currents)-sum(targets), initial=0)

    # This is so inefficent, it loops until the layer sizes are at least as small as the targets
    while True in [a > b for a, b in zip(currents, targets)]:
        for i in layers:
            if currents[i] > targets[i]:
                # Reduce layer i

                # First get layer i+1's state dictionary
                state_dict = model.state_dict_of_layer_i(i+1)

                # check the planned reduction and save it
                reduction = currents[i] - max(targets[i], currents[i] - config("LayerIteration")[i])
                currents[i] = max(targets[i], currents[i] - config("LayerIteration")[i])

                # Get the actual Module from the model, I made a function to get it from the state dictionary key
                old_layer = Imported_Code.get_layer_by_state(model, path_for_layer[i])

                # These need to be unique because I don't know of any generic cloning for the layers
                if isinstance(old_layer, torch.nn.Linear):
                    Imported_Code.set_layer_by_state(model, path_for_layer[i], torch.nn.Linear(old_layer.in_features, currents[i]))
                elif isinstance(old_layer, torch.nn.Conv1d):
                    Imported_Code.set_layer_by_state(model, path_for_layer[i], torch.nn.Conv1d(old_layer.in_channels, currents[i], old_layer.kernel_size))
                else:
                    print(f"Theseus problem, an unknown type of layer just tried to be reduced {old_layer}")
                    raise ValueError()

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

                # update progressbar
                tq.update(reduction)

    tq.close()

    config("PruningSelection", "iteritive_full_theseus")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "config": config}


def addm_test(model: modelstruct.BaseDetectionModel, config: cfg.ConfigObject, **kwargs):
    # Adds the compatability to the model that is needed
    Imported_Code.add_addm_v_layers(model)

    # Reset the optimizer to include the v layers (This is to emulate the original code, not sure if it is in the paper)
    optimizer = model.cfg("Optimizer")(model.parameters(), lr=model.cfg("LearningRate"))

    # Adds an interpretation layer to the config so that it can be read
    wrapped_cfg = Imported_Code.ConfigCompatabilityWrapper(config, translations="ADDM")

    # Expects model to have log_softmax applied (observed from the model used in the code)
    remover = model.register_forward_hook(lambda module, args, output: torch.nn.functional.log_softmax(output, dim=1))

    # Performs the pruning method
    Imported_Code.prune_admm(wrapped_cfg, model, config("Device"), model.dataloader, model.validation_dataloader, optimizer)

    # Applies the pruning to the base model, NOTE: MIGHT CAUSE ISSUES WITH "Imported_Code.remove_addm_v_layers"
    Imported_Code.apply_filter(model, config("Device"), wrapped_cfg)
    # Imported_Code.apply_filter(model, config("Device"), wrapped_cfg)  # Commenting out because I think this is redundent
    mask = Imported_Code.apply_prune(model, config("Device"), wrapped_cfg)
    for m in mask:
        a = mask[m].cpu()
        # I am just making it that the values that need not be updated are nan
        a.apply_(lambda x: x if x == 0 else torch.nan)
        mask[m] = a.to(config("Device"))

    # Calculates the weights that should be kept stable because they are pruned
    # frozen = {name: torch.ones_like(m, requires_grad=False) for name, m in mask.items()}
    # frozen = {name: weight*frozen[name] for name, weight in model.named_parameters() if name in frozen.keys()}
    model.frozen = model.frozen | mask

    print(f"Before retraining: {model.fit()}")
    model.train()
    print(model.fit(epochs=config("NumberOfEpochs")))
    model.eval()

    # Removes the added v layers from the model but ports their values over as a multiplier to the weights
    Imported_Code.remove_addm_v_layers(model)

    # Remove F.log_softmax
    remover.remove()

    model.eval()

    config("PruningSelection", "ADDM_Joint")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "config": config}


def thinet_test_old(config: cfg.ConfigObject, model: modelstruct.BaseDetectionModel, layers: list[int] | None = None, **kwargs):
    if layers is None:
        layers = [layernum for layernum, layerpercent in enumerate(config("WeightPrunePercent")) if layerpercent < 1]

    for i in layers:
        Imported_Code.thinet_pruning(model, i, config=config)

    config("PruningSelection", "thinet_recreation")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "config": config, "layers": layers}


def thinet_test(config: cfg.ConfigObject, model: modelstruct.BaseDetectionModel, layers: list[int] | None = None, **kwargs):
    if layers is None:
        layers = [layernum for layernum, layerpercent in enumerate(config("WeightPrunePercent")) if layerpercent < 1]

    # Create sample of dataset, Fancy load_kwargs is just there to load the collate_fn
    training_data: torch.Tensor = iter(torch.utils.data.DataLoader(model.dataloader.dataset, 100000, **(model.dataloader.base.load_kwargs if hasattr(model.dataloader, "base") else {}))).__next__()[0]
    training_data = training_data.to(model.cfg("Device"))

    for i in layers:
        Imported_Code.run_thinet_on_layer(model, i, training_data=training_data, config=config)

    config("PruningSelection", "thinet")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "config": config, "layers": layers}


def bert_of_theseus_test(model: modelstruct.BaseDetectionModel, config: cfg.ConfigObject, **kwargs):
    # Ok so this whole outside bit is just to figure out what layers we want to replace

    # This splits the model into the predecessors
    module_group_size = int(1/(sum(config("WeightPrunePercent"))/len(config("WeightPrunePercent"))))
    module_group = [x for x in model.get_important_modules()[::module_group_size]]

    hook_object_tuples = []  # This stores hooks for getting the expected successor inputs/outputs
    hook_removers: list[torch.utils.hooks.RemovableHandle] = []  # This stores the unhooking objects for the above

    # List of all modules that are not containers (containers being purely structural and not actually functional)
    lst_of_modules: list[torch.nn.Module] = [m for m in model.modules() if not isinstance(m, modelstruct.container_modules)]
    for start, end in zip(module_group, module_group[1:]):
        # This is for each predecessor, orginized by the starting and ending modules

        # Get a list of all the modules that are encompassed by the predecessor
        lst = lst_of_modules[lst_of_modules.index(start): lst_of_modules.index(end)]

        # Create objects to hold the expected inputs and outputs of the module
        hook_object_tuples.append((Imported_Code.forward_hook(), Imported_Code.forward_hook(), lst))

        # Attach them to the module and then collect the removal objects
        hook_removers.append(lst[0].register_forward_hook(hook_object_tuples[-1][0]))
        hook_removers.append(lst[-1].register_forward_hook(hook_object_tuples[-1][1]))

    # Create sample of dataset, Fancy load_kwargs is just there to load the collate_fn
    training_data = iter(torch.utils.data.DataLoader(model.dataloader.dataset, 100, **(model.dataloader.base.load_kwargs if hasattr(model.dataloader, "base") else {}))).__next__()[0]
    training_data = training_data.to(model.cfg("Device"))

    # Ruh through the small bit of training data to collect module sizes
    model(training_data)

    # Remove those collection hooks from the model
    [x.remove() for x in hook_removers]
    del hook_removers

    # Create the objects that hold the successor modules in the right shape and on the right device
    replace_objects = []
    for start, end, lst in hook_object_tuples:
        replace_objects.append(Imported_Code.Theseus_Replacement(lst, start.inp.shape[1:], end.out.shape[1:], model=model))
        replace_objects[-1].to(config("Device"))

    # Train with the successors/predecessors
    config("PruningSelection", "BERT_theseus_training")
    model.fit(epochs=model.cfg("NumberOfEpochs"))

    # Replace all predecessors with successors, calling it 'condense' because it is removing the clunky hooks
    [x.condense_in_model(model) for x in replace_objects]

    config("PruningSelection", "BERT_theseus")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "config": config, "logger": logger}


def DAIS_test(model: modelstruct.BaseDetectionModel, config: cfg.ConfigObject, layers: list[int] | None = None, **kwargs):
    # Find the layers to apply it to
    if layers is None:
        layers = [layernum for layernum, layerpercent in enumerate(config("WeightPrunePercent")) if layerpercent < 1]

    alphas = []
    for layer, module in enumerate(model.get_important_modules()):
        if layer in layers:
            alphas.append(Imported_Code.add_alpha(module, config("WeightPrunePercent")[layer], config("NumberOfEpochs"), lasso=config("LassoForDAIS")))

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

    return kwargs | {"model": model, "config": config, "logger": logger}


def TOFD_test(model: modelstruct.BaseDetectionModel, config: cfg.ConfigObject, layers: list[int] | None = None, **kwargs):
    wrap = Imported_Code.task_oriented_feature_wrapper(model)

    optimizer = config("Optimizer")(wrap.parameters(), lr=config("LearningRate"))
    wrap.fit(model.dataloader, optimizer, config("NumberOfEpochs"))

    args = Imported_Code.ConfigCompatabilityWrapper(config=config, translations="TOFD")
    new_net = modelstruct.getModel(config)

    # Creating new state dict
    new_net_state_dict = {}
    prior_percentage = 1
    for module_num, module in enumerate(model.get_important_modules()):
        state_dict = new_net.state_dict_of_layer_i(module_num)
        percentage = config("WeightPrunePercent")[module_num]
        new_net_state_dict.update({x: torch.rand_like(y[:math.ceil(len(y)*percentage), :math.ceil(len(y[0])*prior_percentage)]) for x, y in state_dict.items() if ("weight" in x)})
        new_net_state_dict.update({x: torch.rand_like(y[:math.ceil(len(y)*percentage)]) for x, y in state_dict.items() if ("bias" in x)})
        prior_percentage = percentage

    # print(*[f"{x}:{y.shape}\n" for x, y in model.state_dict().items()])
    # print(*[f"{x}:{y.shape}\n" for x, y in new_net_state_dict.items()])
    new_net.load_model_state_dict_with_structure(new_net_state_dict)
    # RESET PARAMETERS FOR NEW SIZE. Found here: https://discuss.pytorch.org/t/reset-model-weights/19180/6
    for layer in new_net.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    new_net.to(model.cfg("Device"))

    # Model set up
    new_net.set_training_data(model.dataloader)
    new_net.set_validation_data(model.validation_dataloader)
    new_net.cfg = config

    new_wrap = Imported_Code.task_oriented_feature_wrapper(new_net)

    optimizer = new_net.cfg("Optimizer")(new_wrap.parameters(), lr=config("LearningRate"))

    # This is the actual TOFD run, everything else is just setup (create aux modules/create student model)
    new_wrap = Imported_Code.TOFD_name_main(optimizer=optimizer, teacher=wrap, net=new_wrap, trainloader=model.dataloader, testloader=model.validation_dataloader, device=config("Device"), args=args, epochs=config("NumberOfEpochs"), LR=config("LearningRate"), criterion=config("LossFunction")())

    wrap.remove()
    new_wrap.remove()

    new_net = new_wrap.wrapped_module

    config("PruningSelection", "TOFD")

    logger = filemanagement.ExperimentLineManager(cfg=config)

    new_net.train(False)

    return kwargs | {"model": new_net, "config": config, "logger": logger}


def Random_test(model: modelstruct.BaseDetectionModel, config: cfg.ConfigObject, **kwargs):
    pruning_list = []
    for count, module in enumerate(model.get_important_modules()):
        pruning_layer = (extramodules.PostMutablePruningLayer(module, register_parameter=False))
        n = len(pruning_layer.para)
        random_permutation = torch.randperm(n, device=model.cfg("Device"))
        random_filter: torch.Tensor = random_permutation.less((config("WeightPrunePercent")[count]*n)//1)
        pruning_layer.para.data = random_filter.type_as(pruning_layer.para.data)
        pruning_list.append(pruning_layer)

    # Retrain after removing layers
    model.fit(config("NumberOfEpochs"))

    for pruning_layer in pruning_list:
        pruning_layer.remove(update_weights=True)

    model.train(False)

    config("PruningSelection", "RandomStructured")

    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "config": config, "logger": logger}


def Recreation_run(model: modelstruct.BaseDetectionModel, config: cfg.ConfigObject, layers: list[int] | None = None, **kwargs):
    # This just makes a new model with the specified size
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
        targets.append(math.ceil(len(state_dict[weights_path])*(1-(1-config("WeightPrunePercent")[i])/2)))  # find how small it should be pruned to
        currents.append(len(state_dict[weights_path]))  # save the shape it currently is

    # This is so inefficent, it loops until the layer sizes are at least as small as the targets
    for i in layers:
        if currents[i] > targets[i]:
            # Reduce layer i

            # Get the actual Module from the model, I made a function to get it from the state dictionary key
            old_layer = Imported_Code.get_layer_by_state(model, path_for_layer[i])

            # These need to be unique because I don't know of any generic cloning for the layers
            if isinstance(old_layer, torch.nn.Linear):
                Imported_Code.set_layer_by_state(model, path_for_layer[i], torch.nn.Linear(old_layer.in_features, targets[i]))
            elif isinstance(old_layer, torch.nn.Conv1d):
                Imported_Code.set_layer_by_state(model, path_for_layer[i], torch.nn.Conv1d(old_layer.in_channels, targets[i], old_layer.kernel_size))
            else:
                raise ValueError()

            # Reduce layer i+1 input
            state_dict = model.state_dict_of_layer_i(i+1)
            weights_path = [x for x in state_dict.keys() if "weight" in x][0]

            # Get the actual Module from the model, I made a function to get it from the state dictionary key
            old_layer = Imported_Code.get_layer_by_state(model, weights_path)

            # These need to be unique because I don't know of any generic cloning for the layers
            if isinstance(old_layer, torch.nn.Linear):
                Imported_Code.set_layer_by_state(model, weights_path, torch.nn.Linear(targets[i], old_layer.out_features))
            elif isinstance(old_layer, torch.nn.Conv1d):
                Imported_Code.set_layer_by_state(model, weights_path, torch.nn.Conv1d(targets[i], old_layer.out_channels, old_layer.kernel_size))
            else:
                raise ValueError()

    config("PruningSelection", "Recreation_Run")
    logger = filemanagement.ExperimentLineManager(cfg=config)

    return kwargs | {"model": model, "logger": logger, "config": config}


types_of_tests = {
    "ADDM_Joint": addm_test,
    "thinet_recreation": thinet_test_old,
    "thinet": thinet_test,
    "iteritive_full_theseus": swapping_run,
    "BERT_theseus": bert_of_theseus_test,
    "DAIS": DAIS_test,
    "TOFD": TOFD_test,
    "RandomStructured": Random_test,
    "Reduced_Normal_Run": Recreation_run,
    "1": addm_test,
    "2": TOFD_test,
    "3": swapping_run,
    "4": bert_of_theseus_test,
    "5": DAIS_test,
    "6": thinet_test,
    "7": Random_test,
    "8": Recreation_run
}
