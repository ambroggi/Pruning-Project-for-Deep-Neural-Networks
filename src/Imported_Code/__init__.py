import sys
import torch

from ..extramodules import PostMutablePruningLayer
# import os

# for subpackage in glob.glob("src/Imported_Code"):
#     print(f"Adding \'{subpackage}\' to system path")
#     sys.path.append(subpackage)

from .helperFunctions import collect_module_is, remove_layers, run_one_channel_module, get_layer_by_state, set_layer_by_state, forward_hook
collect_module_is, remove_layers, run_one_channel_module, get_layer_by_state, set_layer_by_state, forward_hook

sys.path.append("src/Imported_Code/admm_joint_pruning/joint_pruning")
from .admm_joint_pruning.joint_pruning.main import prune_admm, apply_filter, apply_prune, apply_l1_prune
prune_admm, apply_filter, apply_prune, apply_l1_prune
sys.path.remove("src/Imported_Code/admm_joint_pruning/joint_pruning")

from .ThiNet_From_Paper import thinet_pruning
thinet_pruning

from .ThiNet_From_Code import value_sum, value_sum_another
value_sum, value_sum_another

from .BERT_Theseus_From_Paper import Theseus_Replacement
Theseus_Replacement

from .DAIS_From_Paper import add_alpha, DAIS_fit
add_alpha, DAIS_fit

class ConfigCompatabilityWrapper():
    def __init__(self, config):
        self.config = config

    def __getattr__(self, name: str):
        transaltions = {
            "num_epochs": "NumberOfEpochs",
            "lr": "LearningRate",
            "num_pre_epochs": "NumberOfEpochs",
            "alpha": "AlphaForADMM",
            "rho": "RhoForADMM",
            "k": "LayerPruneTargets",
            "percent": "WeightPrunePercent"
        }
        if name in ["l2"]:
            return True
        if name in ["l1"]:
            return False

        return self.config(transaltions.get(name, name))


def add_addm_v_layers(model: torch.nn.Module):
    count = 1
    for module in model.modules():
        print(module)
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv1d):
            model.pruning_layers.append(PostMutablePruningLayer(module))
            model.register_parameter(f"v{count}", model.pruning_layers[-1].para)
            count += 1


def remove_addm_v_layers(model: torch.nn.Module):
    count = 1
    while len(model.pruning_layers) > 0:
        model.__setattr__(f"v{count}", None)
        model.pruning_layers.pop().remove()
        count += 1


def run_thinet_on_layer(model: torch.nn.Module, layerIndex: int, training_data, config):
    module_results: list[forward_hook] = collect_module_is(model, [layerIndex, layerIndex+1], training_data)
    x = torch.stack(run_one_channel_module(module_results[0].modu, module_results[0].inp.detach())).detach().T
    y = module_results[1].out_no_bias.detach()

    if (y.ndim > 2 and y.shape[1] != 1):
        y = torch.sum(y, dim=-1)
    elif y.shape[1] == 1:
        y = torch.flatten(y, start_dim=1)

    indexes, weight_mod = value_sum(x, y, config("WeightPrunePercent")[layerIndex-1])

    keep_tensor = torch.zeros_like(x[0], dtype=torch.bool)
    keep_tensor[indexes] = True

    if module_results[1].modu.weight.data.ndim == 3:
        # This is for CNN layers
        module_results[1].modu.weight.data[:, keep_tensor, :] *= weight_mod.T[:, :, None]
    elif module_results[1].modu.weight.data.ndim == 2:
        # This is for fully connected layers
        if len(keep_tensor) != len(module_results[1].modu.weight.data[0]):
            # This is just for a transition layer from CNN to FC
            multiplier = len(module_results[1].modu.weight.data[0])//len(keep_tensor)
            t2 = torch.zeros(len(module_results[1].modu.weight.data[0]), dtype=torch.bool)
            for i in range(len(keep_tensor)):
                t2[i*multiplier:(i+1)*multiplier] = keep_tensor[i]
        else:
            module_results[1].modu.weight.data[:, keep_tensor] *= weight_mod.T

    remove_layers(model, layerIndex, keepint_tensor=keep_tensor)


print("test")
