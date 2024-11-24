import datetime
import fcntl
import os
import platform
import time

import pandas as pd
import torch

from .cfg import ConfigObject


class ExperimentLineManager():
    def __init__(self, pth: str | os.PathLike = "results/record.csv", cfg: ConfigObject | None = None):
        self.pth = pth
        if pth == "results/record.csv" and cfg is not None and cfg("ResultsPath") is not None:
            self.pth = cfg("ResultsPath")

        # Check that config exists
        if cfg is None:
            # Import is here because things should give a config when using the ExperimentLineManager
            self.cfg = ConfigObject()
            self.cfg("Notes", self.cfg("Notes") | 2)    # Bit 2 of notes is that the value may be inaccurate
        else:
            self.cfg = cfg

        # Create the dataframe
        df = pd.DataFrame(self.cfg.parameters, columns=self.cfg.parameters.keys())[:1]
        df["ProcessID"] = [os.getpid()]
        df["StartTime"] = datetime.datetime.now()
        df["cpuModel"] = platform.processor()
        df["gpuModel"] = torch.cuda.get_device_name() if self.cfg("Device").type == "cuda" else "none"
        df["gpuInfo"] = str(torch.cuda.get_device_properties(torch.cuda.current_device())) if self.cfg("Device").type == "cuda" else "none"
        # print(df["StartTime"])

        # Attach the history
        if os.path.exists(self.pth):
            # File locking so that data is not overridden: https://www.geeksforgeeks.org/file-locking-in-python/
            with open(self.pth, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                # Actual writing:
                hist = pd.read_csv(self.pth, index_col=0)
                # time.sleep(10)
                df = pd.concat([hist, df], ignore_index=True)
                df.to_csv(self.pth)

                # File unlocking
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        else:
            with open(self.pth, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                df.to_csv(self.pth)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self.row_id = df.last_valid_index()
        self.pid = os.getpid()
        print(f"Started row {self.row_id} of file {self.pth}, in process {self.pid}")

    def add_measure(self, measure_name: str, val, **kwargs):
        """
        Adds a value to the history log for the row. It uses flock to lock the file while writing.
        Notably the use of flock means that this function hangs if the file is open with Excel.
        If that cell has already been written to, "Notes" is added and the third bit is set to mark that an overwrite has occurred.

        Args:
            measure_name (str): The name of the column to add the value to. This creates the column if it does not exist.
            val (_type_): Value to add to the table cell.
            can_overwrite (bool): This added measure can overwrite an existing measure without triggering notes to change.

        Raises:
            FileChangedError: The file changed so that the processid for the row no longer matches with this process's pid.
            This indicates that the flock did not work and that the values are not necessarily correctly correlated.
        """
        if isinstance(val, list):
            for n, a in enumerate(val):
                self.add_measure(f"{measure_name}_{n}", a)

        with open(self.pth, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            hist = pd.read_csv(self.pth, index_col=0, )
            # time.sleep(10)

            # Check that the file still has the same process id attached
            if hist.at[self.row_id, "ProcessID"] != self.pid:
                raise FileChangedError()

            # Check if that space is empty, if it is not an error may have occurred
            if (measure_name in hist and not pd.isnull(hist.at[self.row_id, measure_name])) and not kwargs.get("can_overwrite", False):
                hist.at[self.row_id, "Notes"] = hist.at[self.row_id, "Notes"] | 4  # Bit 4 of notes is that an overwrite has occurred

            if isinstance(val, list):
                val = str(val)

            if isinstance(val, str) and hasattr(hist, measure_name) and not isinstance(hist[measure_name].dtype,  pd.StringDtype):
                hist[measure_name] = hist[measure_name].astype(str)

            hist.at[self.row_id, measure_name] = val
            hist.to_csv(self.pth)

            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def __call__(self, measure_name: str, val, **kwargs):
        self.add_measure(measure_name=measure_name, val=val, **kwargs)

    def add_dict(self, dictionary_: dict[str, any]):
        for name, val in dictionary_.items():
            self(name, val)


def load_cfg(pth: str | os.PathLike = "results/record.csv", row_number=-1, config: ConfigObject | None = None, structure_only=True) -> ConfigObject:
    """
    Loads the config from a given history row. This sets the config up so that the specific model can be loaded back.
    If structure_only is true, it only loads the structure config values leaving the rest unchanged, otherwise it tries to overwrite the whole config

    Args:
        pth (str | os.PathLike, optional): Path to load from, should be a history/log csv. Defaults to "results/record.csv".
        row_number (int, optional): Row number to load, if -1 then the last row of the table is loaded. Defaults to -1.
        config (ConfigObject | None, optional): Config object to overwrite, if None, a config is generated. Defaults to None.
        structure_only (bool, optional): If true, only loads the necessarily config values and leaves the rest alone. Defaults to True.

    Returns:
        ConfigObject: This should be the same config object that was passed in, but with overridden values to match the selected config line.
    """
    config = ConfigObject() if config is None else config

    single_row, row_number = history_row(pth, row_number)

    for entry_name in single_row.keys():
        if structure_only:
            # Only loads the structure
            if entry_name in config.structuralOnly:
                config(entry_name, single_row[entry_name])
        else:
            # Loads everything that is not nailed down
            if entry_name in config.parameters.keys() and entry_name not in config.readOnly:
                config(entry_name, single_row[entry_name])

    return config, row_number


def history_row(pth: str | os.PathLike = "results/record.csv", row_number=-1) -> tuple[pd.Series, int]:
    """
    Reads a given row from the history/log file and returns it as a tuple of a pandas series and an integer where the integer is the row index number.
    Note that this is a slow function because it both uses flock to make sure the file is not being edited at the time and it reads in the whole file before selecting the specific row.

    Args:
        pth (str | os.PathLike, optional): Path to the log file. Defaults to "results/record.csv".
        row_number (int, optional): Row that is going to be loaded, if -1 it loads the last row of the file. Defaults to -1.

    Returns:
        tuple[pd.Series, int]: row values in a pandas series as well as row number. The row number is returned for when row_number == -1
    """
    with open(pth, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        # Actual writing:
        hist = pd.read_csv(pth, index_col=0, )

        # File unlocking
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    hist = hist.fillna(value="None")

    if row_number < 0:
        row_number += hist.last_valid_index() + 1

    single_row = hist.iloc[row_number]

    return single_row, row_number


class FileChangedError(Exception):
    def __init__(self, **kwargs):
        super.__init__(kwargs)
        self.message = "File changed from when this row was created. It is no longer valid."


if __name__ == "__main__":
    import random
    e = ExperimentLineManager()
    x = random.randint(0, 1000)
    time.sleep(random.randint(0, 15))
    e("Testing", x)
    print(f"{os.getpid()}:{x}")
