import argparse
import os
import sys
from typing import Literal

import git
import numpy
import torch
import torch.optim

# Printed out by print(config.parameters.keys()) and just copied here, just used for type hints.
CONFIG_OPTIONS = Literal['Version', 'Notes', 'LossFunction', 'Optimizer', 'LearningRate', 'SchedulerLR', 'NumberOfEpochs', 'ModelStructure', 'Dropout', 'HiddenDim', 'HiddenDimSize', 'DatasetName', 'BatchSize', 'TrainTest', 'MaxSamples', 'Dataparallel', 'NumberOfWorkers', 'Device', 'AlphaForADMM', 'RhoForADMM', 'LayerPruneTargets', 'WeightPrunePercent', 'PruningSelection', 'BERTTheseusStartingLearningRate', 'BERTTheseusLearningRateModifier', 'AlphaForTOFD', 'BetaForTOFD', 'tForTOFD', 'DAISRegularizerScale', 'LassoForDAIS', 'LayerIteration', 'TheseusRequiredGrads', 'SaveLocation', 'FromSaveLocation', 'NumClasses', 'NumFeatures', 'NumberWeightSplits', 'ResultsPath']


def get_version() -> str:
    """
    This function just tries to estimate the version from the git history. Tags are considered to be releases and commits are version numbers.

    Returns:
        str: The version number in the form "A.B - C" Where A is the version number, B is the commits since that version, and C is the total commits.
    """
    try:
        repo = git.Repo(os.getcwd())
        # print(f"Tags: {repo.tags}")
        commit_count = len([1 for _ in repo.iter_commits()])

        best_tag = (0, 0)
        for tag in repo.tags:
            commit_num = tag.commit.count()
            if commit_num > best_tag[1] and commit_num <= commit_count:
                best_tag = (best_tag[0] + 1, commit_num)
                # The version is Vx.y - z,
                # where x is a tagged commit, and y is the number of commits after that, and z is the actual number

        return f"v{best_tag[0]}.{commit_count-best_tag[1]} - {commit_count}"
    except git.InvalidGitRepositoryError:
        "0.0 - 0"


class ConfigObject():
    _command_line_args = False
    # The typeChart is supposed to define the possible values of a row
    # typeChart[""] is supposed to convert the type (3rd item in self.parameters) into an actual type.
    # The rest are supposed to list out the possible values of the associated parameter, and any translations from strings that are needed
    _typeChart: dict[str, dict[str, object | str]] = {
        "": {"str": str, "int": int, "float": float, "strl": (str, list), "strdevice": (str, torch.device), "strn": (str, None)},
        "Optimizer": {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "RMS": torch.optim.RMSprop},
        "LossFunction": {"MSE": torch.nn.MSELoss, "CrossEntropy": torch.nn.CrossEntropyLoss},
        "ModelStructure": {"BasicTest": "BasicTest", "SwappingTest": "SwappingTest", "SimpleCNN": "SimpleCNN", "MainLinear": "MainLinear"},
        "DatasetName": {"RandomDummy": "RandomDummy", "Vulnerability": "Vulnerability", "ACI": "ACI", "ACI_grouped": "ACI_grouped", "ACI_grouped_full_balance": "ACI_grouped_full_balance", "ACI_flows": "ACI_flows"},
        "TheseusRequiredGrads": {"All": "All", "Nearby": "Nearby", "New": "New"},
        "Device": {"cpu": torch.device("cpu"), "cuda": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), torch.device("cpu"): torch.device("cpu"), torch.device("cuda"): torch.device("cuda")}
        }
    _parameters: dict[str, list[any, str, str]] = {
            "Version": [get_version(), "Version Number", "str"],
            "Notes": [0, "This is supposed to store a integer associated with specific notes for this config."
                      "\n0 is no Notes,"
                      "\n1 is manually noted inaccuracy (for filtering bad runs after they ran),"
                      "\n2 is that the logger was not given the config object (so it cannot actually say what the config is),"
                      "\n4 is that an unexpected overwrite has occurred (in the record log, the sanity checker failed),"
                      "\n8 is an extra run (same setup but not all done at the same time, only done manually),"
                      "\n16 is that a model was loaded directly without the associated record line (so the config in the logger is not accurate)", "int"],
            "LossFunction": ["CrossEntropy", "Loss function being used", "str"],
            "Optimizer": ["Adam", "Optimizer being used", "str"],
            "LearningRate": [0.0009, "Learning rate for training", "float"],
            "SchedulerLR": [1, "Wether to use a scheduler for the learning rate or not, 1=use scheduler, 0=Don't use scheduler", "int"],
            "NumberOfEpochs": [150, "Number of epochs used", "int"],
            "ModelStructure": ["MainLinear", "Model structure to use", "str"],
            "Dropout": [0.0002, "Dropout after each hidden layer", "float"],
            "HiddenDim": [27, "Number of hidden dimensions", "int"],
            "HiddenDimSize": [175, "Number of filters in hidden dimension", "int"],
            "DatasetName": ["ACI", "What dataset to use", "str"],
            "BatchSize": [100, "How many samples used per batch", "int"],
            "TrainTest": [0.2, "Fraction of data used in the validation set. Also used for splitting Test from validation.", "float"],
            "MaxSamples": [0, "Maximum number of samples in the data, 0 is no limit. Note that the data is random split", "int"],
            "Dataparallel": [-2, "To use distributed data parallel and if it failed. 0 is off, 1 is active, -1 is failed, -2 is not implemented", "int"],
            "NumberOfWorkers": [0, "Number of worker processes or data-parallel processes if Data-parallel is 1", "int"],
            "Device": ["cuda", "Use CPU or CUDA", "strdevice"],
            "AlphaForADMM": [5e-4, "Alpha value for ADMM model", "float"],
            "RhoForADMM": [1.5e-3, "Rho value for ADMM model", "float"],
            "LayerPruneTargets": ["30, 30, *, 30", "Number of nodes per layer starting with the first layer. NOTE: Will cause an error with ADMM if it is a larger number than the number of filters in that layer", "strl"],
            "WeightPrunePercent": ["0.99, 0.9, *, 1", "Percent of weights to prune down to for each layer", "strl"],
            "PruningSelection": ["", "What pruning was applied", "str"],
            "BERTTheseusStartingLearningRate": [0.5, "What Probability value the Bert Theseus method starts with", "float"],
            "BERTTheseusLearningRateModifier": [0.5, "What k value (equation 6) the Bert Theseus method modifies the probability by (divided by epoch count)", "float"],
            "AlphaForTOFD": [0.05, "Feature distance multiplier, this controls the importance of student-teacher feature similarity in the distillation", "float"],
            "BetaForTOFD": [0.03, "This is the multiplier for orthogonal loss, the higher the number, the more weight orthogonal loss will have", "float"],
            "tForTOFD": [3.0, "'temperature for logit distillation' - description from original code", "float"],
            "DAISRegularizerScale": [2.0, "Multiplier to apply to the loss from having too many nodes", "float"],
            "LassoForDAIS": [False, "Wether to use Lasso or not for DIAS, note that the paper describes the lasso loss but appears to not use it", "int"],
            "LayerIteration": ["10, 30, *, 30", "iterative_full_theseus amount each layer is reduced by in each iteration", "strl"],
            "TheseusRequiredGrads": ["All", "What layers to train with the layer replacement style", "str"],
            "SaveLocation": [None, "What filename the state_dict was saved as, if it was saved at all.", "strn"],
            "FromSaveLocation": [None, "What filename the state_dict was loaded as, if it was loaded at all. "
                                 "Or you can load a specific row by putting an input in the form of 'csv x' where x can be any row number. ex 'csv 5'", "strn"],
            "NumClasses": [-1, "How many classes the model is distinguishing between, -1 is to calculate default", "int"],
            "NumFeatures": [-1, "How many features the model is using, -1 is to calculate default", "int"],
            "NumberWeightSplits": [8, "Purely meta config that determines how many tests to run", "int"],
            "ResultsPath": [None, "Path to put the results csv", "strn"]
        }

    def __init__(self, forceSetParams: dict = {}):
        """
        Creates an maintains the current parameters of the model/run.
        forceSetParams currently does nothing, the plan was to have it override the default parameters, but be different from the From_dict version in that it can also write any values.
        """
        self.readOnly = ["Version", "NumberWeightSplits", "ResultsPath"]
        writeOnce = []  # These are command line args or other things that should only be changed once
        self.structuralOnly = ["ModelStructure", "HiddenDim", "HiddenDimSize", "DatasetName", "NumClasses", "NumFeatures", "SaveLocation", "FromSaveLocation"]

        # Annoying check for mutability to make sure that we don't modify the default values.
        self.parameters = {x: y[0].copy() if isinstance(y[0], list) else y[0] for x, y in self._parameters.items()}

        for paramName, paramValue in forceSetParams.items():
            self.parameters[paramName] = paramValue

        self.writeOnce = []

        # This is for initial setup
        for name, values in self.parameters.items():
            if name in ["PruningSelection"]:
                # Pruning selection is an appending value, so this would not reset it, it would append the same value, which is incorrect.
                self(name, "Reset")
            if name not in self.readOnly:
                self(name, values)

        # Set the values that can only be written once (for reading from args)
        self.writeOnce = writeOnce

    def __call__(self, paramName: str | CONFIG_OPTIONS, paramVal: str | float | int | None = None, getString: bool = False) -> str | float | int | object | None:
        """
        Set or get a parameter from the config.

        Args:
            paramName (str | CONFIG_OPTIONS): The name of the parameter you want to access
            paramVal (str | float | int | None, optional): The value you want to write, leave None if you want to read. Not available on readOnly. Defaults to None.
            getString (bool, optional): _description_. Defaults to False.

        Returns:
            str | float | int | object | None: _description_
        """
        if paramVal is None:
            return self.get_param(paramName, getString=getString)
        else:
            return self.set_param(paramName, paramVal)

    def set_param(self, paramName: str | CONFIG_OPTIONS, paramVal: str | float | int) -> None:
        """
        Sets the given parameter, assuming it is not a readOnly parameter.
        Setting a value to None can be done by setting it to "None".
        Some parameters have unique effects:
            "LayerPruneTargets" and "LayerIteration": Can be entered as a string or a list of ints. Also a star "*" duplicates the value before it N times where N is the current HiddenDim (This is done on read, not write)
            "WeightPrunePercent": Can be entered as a string or a list of floats. Also a star "*" duplicates the value before it N times where N is the current HiddenDim (This is done on read, not write)
            "PruningSelection": Appends new values unless set to "None", so that it might be possible to stack pruning methods

        Args:
            paramName (str | CONFIG_OPTIONS): The parameter to overwrite.
            paramVal (str | float | int): The value to overwrite with.

        Raises:
            ValueError: The paramVal value is not one of the expected possible values for a parameter with discrete values.
            TypeError: The paramVal is not an acceptable type.
        """
        # Deal with numpy types:
        if isinstance(paramVal, numpy.generic):
            paramVal = paramVal.item()
        if paramVal == "None":
            paramVal = "" if "str" in self._parameters[paramName][2] else None

        # These arguments assume None is default
        if paramName in ["NumClasses", "NumFeatures"]:
            if paramVal is None:
                paramVal = self.get_param(paramName)

        # Check if the type is valid by querying typeChart
        if isinstance(paramVal, self.get_param_type(paramName)):
            # Add extra conditionals here:

            if paramName in self._typeChart.keys():  # Check that the value is valid (if applicable)
                if paramVal not in self._typeChart[paramName].keys():
                    raise ValueError(f"{paramName} does not have an option for '{paramVal}'")

            if paramName == "Device":  # Device is translated into a torch.device
                paramVal = self._typeChart["Device"][paramVal]

            if self._parameters[paramName][2] == "strn" and (paramVal == "" or paramVal == "None"):
                paramVal = None

            if paramName in ["LayerPruneTargets", "LayerIteration"]:  # This is a list, so we need to do  a little formatting (ints)
                if isinstance(paramVal, str):
                    paramVal = paramVal.strip("[]")
                    paramVal = [int(x) if (x != "*") else x for x in paramVal.split(", ")]
                else:
                    paramVal = [int(x) if (x != "*") else x for x in paramVal]

            if paramName in ["WeightPrunePercent"]:  # This is a list, so we need to do  a little formatting (floats)
                if isinstance(paramVal, str):
                    paramVal = paramVal.strip("[]")
                    paramVal = [float(x) if (x != "*") else x for x in paramVal.split(", ")]
                else:
                    paramVal = [float(x) if (x != "*") else x for x in paramVal]

            if paramName in ["PruningSelection"]:  # This is supposed to be a running tally, so we need to add it on.
                if not self.get_param(paramName) == "" and not paramVal == "Reset":
                    paramVal = self.get_param(paramName) + "|" + paramVal
                elif paramVal == "Reset":
                    paramVal = ""

            if paramName in self.readOnly:  # Some items should not be able to be modified.
                print(f"Attempted to change config {paramName}, which is Read-Only")
                return

            if paramName in self.writeOnce:  # If they are only allowed to be written to once
                self.writeOnce.remove(paramName)
                self.readOnly.append(paramName)

            # Set the value
            self.parameters[paramName] = paramVal
        else:
            if type(paramVal) is float and int(paramVal) == paramVal:
                return self.set_param(paramName=paramName, paramVal=int(paramVal))
            raise TypeError(f"Attempted to set Config value ({paramName}) of inappropriate type, type={type(paramVal)}, while {paramName} has an expected type(s) of {self.get_param_type(paramName)}")

    def get_param(self, paramName: str | CONFIG_OPTIONS, getString: bool = False) -> str | float | int | object:
        """
        Retrieves a parameter from the config.

        Args:
            paramName (str | CONFIG_OPTIONS): Parameter name to retrieve
            getString (bool, optional): Gets the parameter value as a string, some parameters are converted into objects, if True this skips that step and just returns the string. Defaults to False.

        Returns:
            str | float | int | object: The value of the parameter
        """
        if (paramName in self._typeChart.keys()) and not getString:
            return self._typeChart[paramName][self.parameters[paramName]]

        if paramName in ["LayerPruneTargets", "LayerIteration", "WeightPrunePercent"] and not getString:
            if "*" in self.parameters[paramName]:
                i = self.parameters[paramName].index("*")
                return self.parameters[paramName][:i] + [self.parameters[paramName][i-1] for _ in range(self("HiddenDim") - 1)] + self.parameters[paramName][i+1:]

        return self.parameters[paramName]

    def get_param_description(self, paramName: str | CONFIG_OPTIONS) -> str:
        """
        Gets the description of the parameter, this can also be found in the cfg.py file, this is just here if you want to fetch it during runtime for some reason?

        Args:
            paramName (str | CONFIG_OPTIONS): Parameter you want to get the description of.

        Returns:
            str: A text description of what the parameter is.
        """
        return self._parameters[paramName][1]

    def get_param_type(self, paramName: str | CONFIG_OPTIONS) -> object:
        """
        Gets the types accepted by a parameter when you want to set it.

        Args:
            paramName (str | CONFIG_OPTIONS): The parameter you want to get the possible types of.

        Returns:
            object: type object or tuple of types. (made for passing to isinstance())
        """
        return self._typeChart[""][self._parameters[paramName][2]]

    @classmethod
    def get_param_from_args(cls) -> "ConfigObject":
        """
        This is the project argument parsing method. It reads the arguments and sets those as the defaults for the config object, it also generates a new config object to return back. Unknown arguments are lost.
        Note: does not work with pytest, I think the parsing library has some kind of conflict so this just does not run if pytest is imported, it passes a default ConfigObject.
        Another Note: It only actually parses the arguments on the first run. All future runs it uses the saved value.

        Returns:
            ConfigObject: The ConfigObject generated by the command line arguments of this python program.
        """
        if not cls._command_line_args:
            if "pytest" not in sys.modules:  # The argument parser appears to have issues with the pytest tests. I have no idea why.
                # Argparse tutorial: https://docs.python.org/3/howto/argparse.html
                parser = argparse.ArgumentParser()

                for p in cls._parameters.keys():
                    if p not in ["Notes", "Version"]:
                        if p in cls._typeChart.keys():
                            t = cls._typeChart[""][cls._parameters[p][2]]
                            t = t[0] if isinstance(t, (list, tuple)) else t
                            parser.add_argument(f"--{p}", choices=cls._typeChart[p].keys(), type=t, default=cls._parameters[p][0], help=cls._parameters[p][1], required=False)
                        else:
                            t = cls._typeChart[""][cls._parameters[p][2]]
                            t = t[0] if isinstance(t, (list, tuple)) else t
                            parser.add_argument(f"--{p}", type=t, default=cls._parameters[p][0], help=cls._parameters[p][1], required=False)

                # Parse the args
                args, unknown_args = parser.parse_known_args()
                # args = parser.parse_args()

                # Set the defaults
                for paramName, paramValue in args._get_kwargs():
                    cls._parameters[paramName][0] = paramValue

                cls._command_line_args = True
            else:
                print("Pytest has problems with ArgumentParser")
                assert False, "Should not be using get_param_from_args with pytest"

        self_ = cls()

        if args:
            for paramName, paramValue in args._get_kwargs():
                assert str(self_(paramName, getString=True)) == str(paramValue) or paramName in ["Device", "LayerPruneTargets", "WeightPrunePercent", "LayerIteration"]

        return self_

    def clone(self) -> "ConfigObject":
        """
        Duplicates ConfigObject. It returns a new ConfigObject with everything identical to the original except for readOnly parameters (which are unlikely to change within a single execution of the code)

        Returns:
            ConfigObject: Newly created ConfigObject that should be the same as the old one.
        """
        new = ConfigObject()
        for x in self.parameters:
            if x in ["PruningSelection"]:
                new(x, "Reset")
            if x not in self.readOnly:
                new(x, self(x, getString=True))
        return new

    def to_dict(self) -> dict[str, str]:
        """
        Turns the config into a dictionary so that it can be serialized and passed to another process.

        Returns:
            dict[str, str]: Each parameter represented as a value in a dictionary so that it can be loaded with from_dict()
        """
        return {x: self.get_param(x, getString=True) for x in self.parameters.keys()}

    @classmethod
    def from_dict(cls, dictionary: dict[str, str]) -> "ConfigObject":
        """
        Turns a dictionary back into a ConfigObject. This is used to unpack from serialization.

        Args:
            dictionary (dict[str, str]): A dictionary of as many parameters and their values as you want. All unlisted parameters are left as defaults.

        Returns:
            ConfigObject: The created ConfigObject.
        """
        self_ = cls()
        # This is for initial setup
        for name, value in dictionary.items():
            if name in ["PruningSelection"]:
                self_(name, "Reset")
            if name not in self_.readOnly:
                self_(name, value)

        return self_


def make_version_tag(message: str):
    """
    Unused and not working; function that tags the current version in the git history

    Args:
        message (str): Message to use for the git commit.
    """
    repo = git.Repo(os.getcwd())
    commit_count = len([1 for _ in repo.iter_commits()])
    repo.create_tag(f"Commit: {commit_count}", message=message)
