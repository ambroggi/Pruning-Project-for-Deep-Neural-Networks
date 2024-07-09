import argparse
import sys
import git
import os


class ConfigObject():

    def __init__(self):
        self.typechart = {"str": str, "int": int, "float": float}
        self.parameters = {
            "Version": [get_version(), "Version Number", "str"],
            "Notes": [0, "This is supposed to store a integer associated with specific notes for this config. 0 is no Notes", "int"]
        }

    def __call__(self, paramName: str, paramVal: str | float | int | None = None):
        if paramVal is None:
            return self.get_param(paramName)
        else:
            return self.set_param(paramName, paramVal)

    def set_param(self, paramName: str, paramVal: str | float | int):
        if isinstance(paramVal, self.typechart[self.parameters[paramName][2]]):
            # Add extra conditionals here
            self.parameters[paramName][0] = paramVal
        else:
            raise TypeError("Attempted to set Config value of inappropriate type")

    def get_param(self, paramName: str) -> str | float | int:
        return self.parameters[paramName][0]

    def get_param_description(self, paramName: str) -> str:
        return self.parameters[paramName][1]

    def get_param_type(self, paramName: str) -> object:
        return self.typechart[self.parameters[paramName][2]]

    @staticmethod
    def get_param_from_args():
        self_ = ConfigObject()

        if "pytest" not in sys.modules:  # The argument parser appears to have issues with the pytest tests. I have no idea why.
            # Argparse tutorial: https://docs.python.org/3/howto/argparse.html
            parser = argparse.ArgumentParser()

            for p in self_.parameters.keys():
                parser.add_argument(f"--{p}", type=self_.typechart[self_.parameters[p][2]], default=self_.parameters[p][0], help=self_.parameters[p][1], required=False)

            # Parse the args
            args = parser.parse_args()
            for x in args._get_kwargs():
                self_(x[1], [x[0]][0])
        else:
            print("Pytest has problems with ArgumentParser")
            assert False, "Should not be using get_param_from_args with pytest"

        return self_


def get_version():
    repo = git.Repo(os.getcwd())
    commit_count = len(repo.iter_commits())
    return commit_count
