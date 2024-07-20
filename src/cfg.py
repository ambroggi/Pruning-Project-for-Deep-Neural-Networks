import argparse
import sys
import git
import os
import torch.optim


class ConfigObject():

    def __init__(self):
        self.typechart = {"": {"str": str, "int": int, "float": float},
                          "Optimizer": {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "RMS": torch.optim.RMSprop},
                          "LossFunction": {"MSE": torch.nn.MSELoss, "CrossEntropy": torch.nn.CrossEntropyLoss},
                          "DatasetName": {"RandomDummy": "RandomDummy", "Vulnerability": "Vulnerability"}
                          }

        self.readOnly = ["Version"]

        self.parameters = {
            "Version": [get_version(), "Version Number", "str"],
            "Notes": [0, "This is supposed to store a integer associated with specific notes for this config. 0 is no Notes", "int"],
            "LossFunction": ["CrossEntropy", "Loss function being used", "str"],
            "Optimizer": ["Adam", "Optimizer being used", "str"],
            "LearningRate": [0.0001, "Learning rate for training", "float"],
            "NumberOfEpochs": [10, "Number of epochs used", "int"],
            "DatasetName": ["RandomDummy", "What dataset to use", "str"],
            "BatchSize": [3, "How many samples used per batch", "int"],
            "Dataparallel": [-2, "To use distributed data parallel and if it failed. 0 is off, 1 is active, -1 is failed, -2 is not implemented", "int"],
            "NumberOfWorkers": [0, "Number of worker processes or dataparallel processes if Dataparallel is 1", "int"]
        }

    def __call__(self, paramName: str, paramVal: str | float | int | None = None, getString=False):
        if paramVal is None:
            return self.get_param(paramName, getString=getString)
        else:
            return self.set_param(paramName, paramVal)

    def set_param(self, paramName: str, paramVal: str | float | int):
        if isinstance(paramVal, self.typechart[""][self.parameters[paramName][2]]):
            # Add extra conditionals here
            if paramName in self.typechart.keys():
                if paramVal not in self.typechart[paramName].keys():
                    raise ValueError(f"{paramName} does not have an option for {paramVal}")

            if paramName in self.readOnly:
                print(f"Attempted to change config {paramName}, which is Read-Only")

            # Set the value
            self.parameters[paramName][0] = paramVal
        else:
            raise TypeError("Attempted to set Config value of inappropriate type")

    def get_param(self, paramName: str, getString=False) -> str | float | int:
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
                        parser.add_argument(f"--{p}", choices=self_.typechart[p].keys(), type=self_.typechart[""][self_.parameters[p][2]], default=self_.parameters[p][0], help=self_.parameters[p][1], required=False)
                    else:
                        parser.add_argument(f"--{p}", type=self_.typechart[""][self_.parameters[p][2]], default=self_.parameters[p][0], help=self_.parameters[p][1], required=False)

            # Parse the args
            args = parser.parse_args()
            for paramName, paramValue in args._get_kwargs():
                self_(paramName, paramValue)
        else:
            print("Pytest has problems with ArgumentParser")
            assert False, "Should not be using get_param_from_args with pytest"

        return self_


def get_version():
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
