import argparse
import sys
import git
import os
import torch.optim
import torch
import numpy


class ConfigObject():

    def __init__(self):
        # the typechart is supposed to define the possible values of a row
        # Typechart[""] is supposed to convert the type (3rd item in self.parameters) into an actual type.
        # The rest are supposed to list out the possible values of the associated parameter, and any translations from strings that are needed
        self.typechart: dict[str, dict[str, object | str]] = {
                          "": {"str": str, "int": int, "float": float, "strl": (str, list), "strdevice": (str, torch.device), "strn": (str, None)},
                          "Optimizer": {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "RMS": torch.optim.RMSprop},
                          "LossFunction": {"MSE": torch.nn.MSELoss, "CrossEntropy": torch.nn.CrossEntropyLoss},
                          "ModelStructure": {"BasicTest": "BasicTest", "SwappingTest": "SwappingTest", "SimpleCNN": "SimpleCNN"},
                          "DatasetName": {"RandomDummy": "RandomDummy", "Vulnerability": "Vulnerability"},
                          "Device": {"cpu": torch.device("cpu"), "cuda": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), torch.device("cpu"): torch.device("cpu"), torch.device("cuda"): torch.device("cuda")}
                          }

        self.readOnly = ["Version"]

        self.parameters: dict[str, list[any, str, str]] = {
            "Version": [get_version(), "Version Number", "str"],
            "Notes": [0, "This is supposed to store a integer associated with specific notes for this config. 0 is no Notes", "int"],
            "LossFunction": ["CrossEntropy", "Loss function being used", "str"],
            "Optimizer": ["Adam", "Optimizer being used", "str"],
            "LearningRate": [0.0001, "Learning rate for training", "float"],
            "NumberOfEpochs": [10, "Number of epochs used", "int"],
            "ModelStructure": ["SimpleCNN", "Model structure to use", "str"],
            "DatasetName": ["RandomDummy", "What dataset to use", "str"],
            "BatchSize": [3, "How many samples used per batch", "int"],
            "Dataparallel": [-2, "To use distributed data parallel and if it failed. 0 is off, 1 is active, -1 is failed, -2 is not implemented", "int"],
            "NumberOfWorkers": [8, "Number of worker processes or dataparallel processes if Dataparallel is 1", "int"],
            "Device": ["cuda", "Use CPU or CUDA", "strdevice"],
            "AlphaForADMM": [5e-4, "Alpha value for ADMM model", "float"],
            "RhoForADMM": [1.5e-3, "Rho value for ADMM model", "float"],
            "LayerPruneTargets": ["3, 1, 3, 3", "Number of nodes per layer starting with the first layer. NOTE: Will cause an error with ADMM if it is a larger number than the number of filters in that layer (also used for iteritive_full_theseus)", "strl"],
            "WeightPrunePercent": ["0.72, 0.5, 0.7, 0.8", "Percent of weights to prune down to for each layer", "strl"],
            "PruningSelection": ["", "What pruning was applied", "str"],
            "BERTTheseusStartingLearningRate": [0.5, "What Probibility value the Bert Theseus method starts with", "float"],
            "BERTTheseusLearningRateModifier": [0.5, "What k value (equation 6) the Bert Theseus method modifies the probibility by (devided by epoch count)", "float"],
            "SaveLocation": [None, "What filename the statedict was saved as, if it was saved at all.", "strn"],
            "FromSaveLocation": [None, "What filename the statedict was loaded as, if it was loaded at all.", "strn"]
        }

        # This is for initial setup
        for name, values in self.parameters.items():
            if name not in self.readOnly:
                self(name, values[0])

    def __call__(self, paramName: str, paramVal: str | float | int | None = None, getString: bool = False) -> str | float | int | object | None:
        if paramVal is None:
            return self.get_param(paramName, getString=getString)
        else:
            return self.set_param(paramName, paramVal)

    def set_param(self, paramName: str, paramVal: str | float | int) -> None:
        # Deal with numpy types:
        if isinstance(paramVal, numpy.generic):
            paramVal = paramVal.item()
        if paramVal == "None":
            paramVal = "" if "str" in self.parameters[paramName][2] else None

        # Check if the type is valid by querrying typechart
        if isinstance(paramVal, self.typechart[""][self.parameters[paramName][2]]):
            # Add extra conditionals here:

            if paramName in self.typechart.keys():  # Check that the value is valid (if applicable)
                if paramVal not in self.typechart[paramName].keys():
                    raise ValueError(f"{paramName} does not have an option for '{paramVal}'")

            if paramName == "Device":  # Device is translated into a torch.device
                paramVal = self.typechart["Device"][paramVal]

            if self.parameters[paramName][2] == "strn" and (paramVal == "" or paramVal == "None"):
                paramVal = None

            if paramName in ["LayerPruneTargets"]:  # This is a list, so we need to do  a little formatting (ints)
                if isinstance(paramVal, str):
                    paramVal = paramVal.strip("[]")
                    paramVal = [int(x) for x in paramVal.split(", ")]

            if paramName in ["WeightPrunePercent"]:  # This is a list, so we need to do  a little formatting (floats)
                if isinstance(paramVal, str):
                    paramVal = paramVal.strip("[]")
                    paramVal = [float(x) for x in paramVal.split(", ")]

            if paramName in ["PruningSelection"]:  # This is supposed to be a running tally, so we need to add it on.
                if not self.get_param(paramName) == "":
                    paramVal = self.get_param(paramName) + "|" + paramVal

            if paramName in self.readOnly:  # Some items should not be able to be modified.
                print(f"Attempted to change config {paramName}, which is Read-Only")
                return

            # Set the value
            self.parameters[paramName][0] = paramVal
        else:
            raise TypeError(f"Attempted to set Config value of inappropriate type, type={type(paramVal)}")

    def get_param(self, paramName: str, getString: bool = False) -> str | float | int | object:
        if (paramName in self.typechart.keys()) and not getString:
            return self.typechart[paramName][self.parameters[paramName][0]]
        return self.parameters[paramName][0]

    def get_param_description(self, paramName: str) -> str:
        return self.parameters[paramName][1]

    def get_param_type(self, paramName: str) -> object:
        return self.typechart[""][self.parameters[paramName][2]]

    @staticmethod
    def get_param_from_args():
        self_ = ConfigObject()

        if "pytest" not in sys.modules:  # The argument parser appears to have issues with the pytest tests. I have no idea why.
            # Argparse tutorial: https://docs.python.org/3/howto/argparse.html
            parser = argparse.ArgumentParser()

            for p in self_.parameters.keys():
                if p not in ["Notes"] and p not in self_.readOnly:
                    if p in self_.typechart.keys():
                        t = self_.typechart[""][self_.parameters[p][2]]
                        t = t[0] if isinstance(t, (list, tuple)) else t
                        parser.add_argument(f"--{p}", choices=self_.typechart[p].keys(), type=t, default=self_.parameters[p][0], help=self_.parameters[p][1], required=False)
                    else:
                        t = self_.typechart[""][self_.parameters[p][2]]
                        t = t[0] if isinstance(t, (list, tuple)) else t
                        parser.add_argument(f"--{p}", type=t, default=self_.parameters[p][0], help=self_.parameters[p][1], required=False)

            # Parse the args
            args, unknown_args = parser.parse_known_args()
            # args = parser.parse_args()
            for paramName, paramValue in args._get_kwargs():
                self_(paramName, paramValue)
        else:
            print("Pytest has problems with ArgumentParser")
            assert False, "Should not be using get_param_from_args with pytest"

        return self_

    def clone(self):
        new = ConfigObject()
        for x in self.parameters:
            if x not in self.readOnly:
                new(x, self(x, getString=True))
        return new


def get_version() -> str:
    repo = git.Repo(os.getcwd())
    # print(f"Tags: {repo.tags}")
    commit_count = len([1 for _ in repo.iter_commits()])

    best_tag = (0, 0)
    for tag in repo.tags:
        commit_num = tag.path.split(sep=": ")[1]
        if commit_num > best_tag[1] and commit_num < commit_count:
            best_tag = (best_tag[0] + 1, commit_num)
            # TODO: Make sure this works, I want it so that the version is Vx.y - z,
            # where x is a tagged commit, and y is the number of commits after that, and z is the actual number

    return f"v{best_tag[0]}.{commit_count-best_tag[1]} - {commit_count}"


def make_versiontag(message: str):
    repo = git.Repo(os.getcwd())
    commit_count = len([1 for _ in repo.iter_commits()])
    repo.create_tag(f"Commit: {commit_count}", message=message)
