import glob
import sys
import torch
import torch.nn.utils.prune as prune
# import os


for subpackage in glob.glob("Imported_Code"):
    print(f"Adding \'{subpackage}\' to system path")
    sys.path.append(subpackage)


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
        if name in ["l1", "l2"]:
            return True

        return self.config(transaltions.get(name, name))


def add_addm_v_layers(model: torch.nn.Module):
    count = 1
    for module in model.modules():
        print(module)
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            vmask = torch.nn.Parameter(torch.ones(len(module.weight)), requires_grad=False)
            ADDM_V_Layer.apply(module, "weight", vmask=vmask)
            model.__setattr__(f"v_{count}", vmask)
            count += 1


def remove_addm_v_layers(model: torch.nn.Module):
    count = 1
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.remove(module, "weight")
            model.__setattr__(f"v_{count}", None)
            count += 1


class ADDM_V_Layer(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'

    def __init__(self, vmask):
        self.vmask = vmask

    def compute_mask(self, t, default_mask):
        return default_mask * self.vmask[:, None]  # https://stackoverflow.com/a/53988549


sys.path.append("src/Imported_Code/admm_joint_pruning/joint_pruning")
from .admm_joint_pruning.joint_pruning.main import prune_admm, apply_filter, apply_prune, apply_l1_prune
prune_admm
sys.path.remove("src/Imported_Code/admm_joint_pruning/joint_pruning")
print("test")
