import glob
import sys
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
            "rho": "RhoForADMM"
        }
        if name == "l1":
            return True

        return self.config(transaltions.get(name, name))


sys.path.append("src/Imported_Code/admm_joint_pruning/joint_pruning")
from .admm_joint_pruning.joint_pruning.main import prune_admm
prune_admm
sys.path.remove("src/Imported_Code/admm_joint_pruning/joint_pruning")
print("test")
