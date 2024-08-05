import glob
import sys
import torch

from .. import modelstruct
# import os

for subpackage in glob.glob("Imported_Code"):
    print(f"Adding \'{subpackage}\' to system path")
    sys.path.append(subpackage)


sys.path.append("src/Imported_Code/admm_joint_pruning/joint_pruning")
from .admm_joint_pruning.joint_pruning.main import prune_admm, apply_filter, apply_prune, apply_l1_prune
prune_admm, apply_filter, apply_prune, apply_l1_prune
sys.path.remove("src/Imported_Code/admm_joint_pruning/joint_pruning")

from .ThiNet_From_Paper import thinet_pruning
thinet_pruning


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
            model.pruning_layers.append(modelstruct.PostMutablePruningLayer(module))
            model.register_parameter(f"v{count}", model.pruning_layers[-1].para)
            count += 1


def remove_addm_v_layers(model: torch.nn.Module):
    count = 1
    while len(model.pruning_layers) > 0:
        model.__setattr__(f"v{count}", None)
        model.pruning_layers.pop().remove()
        count += 1


print("test")
