# This is just some code to specifically run ThiNet, I did not want it in the ThiNet_From_Code file because it is not from the original code
# This just handles most of the adaptation between the systems.
from typing import TYPE_CHECKING

import torch
from numpy.linalg import LinAlgError

if TYPE_CHECKING:
    from src.modelstruct import BaseDetectionModel

from .helperFunctions import forward_hook, remove_layers
from .ThiNet_From_Code import value_sum, value_sum_another

value_sum, value_sum_another


def run_one_channel_module(module: torch.nn.Module, dat: torch.Tensor) -> list[torch.Tensor]:
    """
    This runs the module once for each channel with all other channels being zeroed out.
    Then sums up the output for that one channel and outputs these sums as a list.

    So it outputs a list of values the same length as the number of channels that represents the activation for that channel.
    (This is used for ThiNet)
    """
    state_dict = {a: b.clone() for a, b in module.state_dict().items()}
    lst = []
    for x in range(len(state_dict["weight"])):
        state_dict_clone = {a: b.clone() for a, b in state_dict.items()}
        state_dict_clone["bias"] = torch.zeros_like(state_dict_clone["bias"])
        state_dict_clone["weight"] = torch.zeros_like(state_dict_clone["weight"])
        state_dict_clone["weight"][x] = state_dict["weight"][x]

        module.load_state_dict(state_dict_clone)
        out = module(dat)
        while out.ndim > 1:
            out = torch.sum(out, dim=1)
        lst.append(out)

    module.load_state_dict(state_dict)
    return lst


def collect_module_is(model: "BaseDetectionModel", paramNumbers: list, batch: torch.Tensor) -> list[forward_hook]:
    """
    Thinet calls the outputs of the layers i in the code, so this is collect the i s.
    That is, this code creates the hooks for the model that collect the input and output, runs the model, and removes the hooks while passing back the results.

    Args:
        model (BaseDetectionModel): Model to add the hooks to.
        paramNumbers (list): Specific layer numbers to add hooks to.
        batch (torch.Tensor): Batch of data to pass through the model and collect.

    Returns:
        list[forward_hook]: List of hook objects that have all the information about the layer(s).
    """
    # Used in ThiNet
    i = 0
    hooks = []
    removers = []
    for module in model.get_important_modules():
        if i in paramNumbers:
            # Prepare to catch the data for the module
            hooks.append(forward_hook())
            removers.append(module.register_forward_hook(hooks[-1]))
        i += 1

    model(batch)

    for r in removers:
        r.remove()

    return hooks


def run_thinet_on_layer(model: "BaseDetectionModel", layerIndex: int, training_data, config):
    """
    Runs the thinet method on one specific layer.

    Args:
        model (BaseDetectionModel): Model to prune.
        layerIndex (int): Layer to prune by index.
        training_data (_type_): The data to use to prune the model
        config (_type_): Config to use

    Raises:
        e: It sometimes throws a numpy linear algebra error about the inverse of a matrix not existing.
        I think this happens due to the training data not having the correct samples, so I just upped the count for training data.
    """
    module_results: list[forward_hook] = collect_module_is(model, [layerIndex, layerIndex+1], training_data)
    x = torch.stack(run_one_channel_module(module_results[0].modu, module_results[0].inp.detach())).detach().cpu().T
    y = module_results[1].out_no_bias.detach().cpu()

    # Flattening data (the algorithm expects one dimensional data)
    if (y.ndim > 2):
        y = torch.sum(y, dim=-1)
    elif y.shape[1] == 1:
        y = torch.flatten(y, start_dim=1)

    # catch for an edge case
    if x.shape[1] == 1:
        print("Thinet does not make sense on layers with a single input channel")
        remove_layers(model, layerIndex, keepint_tensor=torch.ones_like(x[0], dtype=torch.bool))
        return

    try:
        # This is actually running the thinet algorithm code and getting out the indexes.
        indexes, weight_mod = value_sum(x, y, config("WeightPrunePercent")[layerIndex])
    except LinAlgError as e:
        if not ("Singular matrix" in e.args[0]):
            raise e
        else:
            print("Thinet got a Singular matrix, the reason for this is currently unknown")
    else:
        # Weight_mod is in the shape of the newly created weights for module_results
        # That means it is C_in*C_out where C_in is the number of channels being kept (len(indexes))
        # and C_out is the number of channels the layer used to have

        keep_tensor = torch.zeros_like(x[0], dtype=torch.bool, device=model.cfg("Device"))
        keep_tensor[indexes] = True

        weight_mod = torch.Tensor(weight_mod).to(model.cfg("Device"))

        # Apply the extra weight modification to compensate for the removed nodes.
        if module_results[1].modu.weight.data.ndim == 3:
            # This is for CNN layers
            module_results[1].modu.weight.data[:, keep_tensor, :] *= weight_mod.T[:, :, None]
        elif module_results[1].modu.weight.data.ndim == 2:
            # This is for fully connected layers
            if len(keep_tensor) != len(module_results[1].modu.weight.data[0]):
                # This is just for a transition layer from CNN to FC
                multiplier = len(module_results[1].modu.weight.data[0])//len(keep_tensor)
                t2 = torch.zeros(len(module_results[1].modu.weight.data[0]), dtype=torch.bool)
                w2 = torch.zeros((multiplier*len(weight_mod), len(weight_mod[0])), dtype=torch.float)
                for i in range(len(keep_tensor)):
                    t2[i*multiplier:(i+1)*multiplier] = keep_tensor[i]
                for i in range(len(weight_mod)):
                    w2[i*multiplier:(i+1)*multiplier] += weight_mod[i]

                module_results[1].modu.weight.data[:, t2] *= w2.T
            else:
                module_results[1].modu.weight.data[:, keep_tensor] *= weight_mod.T

        # Apply the removal of the layers
        remove_layers(model, layerIndex, keepint_tensor=keep_tensor)
