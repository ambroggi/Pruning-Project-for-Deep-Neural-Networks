import pandas as pd
import os
import fcntl
import time
import datetime
import platform
from .cfg import ConfigObject


class ExperimentLineManager():
    def __init__(self, pth: str | os.PathLike = "results/record.csv", cfg: ConfigObject | None = None):
        self.pth = pth

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
        # print(df["StartTime"])

        # Attach the history
        if os.path.exists(pth):
            # File locking so that data is not overriden: https://www.geeksforgeeks.org/file-locking-in-python/
            with open(pth, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                # Actual writing:
                hist = pd.read_csv(pth, index_col=0)
                # time.sleep(10)
                df = pd.concat([hist, df], ignore_index=True)
                df.to_csv(pth)

                # File unlocking
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        else:
            with open(pth, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                df.to_csv(pth)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self.row_id = df.last_valid_index()
        self.pid = os.getpid()

    def add_measure(self, measure_name: str, val, **kwargs):
        if isinstance(val, list):
            for n, a in enumerate(val):
                self.add_measure(f"{measure_name}_{n}", a)

        with open(self.pth, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            hist = pd.read_csv(self.pth, index_col=0)
            # time.sleep(10)

            # Check that the file still has the same process id attached
            if hist.at[self.row_id, "ProcessID"] != self.pid:
                raise FileChangedError()

            # Check if that space is empty, if it is not an error may have occured
            if (measure_name in hist and not pd.isnull(hist.at[self.row_id, measure_name])) and not kwargs.get("can_overwrite", False):
                hist.at[self.row_id, "Notes"] = hist.at[self.row_id, "Notes"] | 4  # Bit 4 of notes is that an overwrite has occured

            hist.at[self.row_id, measure_name] = val
            hist.to_csv(self.pth)

            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def __call__(self, measure_name: str, val, **kwargs):
        self.add_measure(measure_name=measure_name, val=val, **kwargs)

    def add_dict(self, dictionary_: dict[str, any]):
        for name, val in dictionary_.items():
            self(name, val)


def load_cfg(pth: str | os.PathLike = "results/record.csv", row_number=-1) -> ConfigObject:
    config = ConfigObject()

    with open(pth, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        # Actual writing:
        hist = pd.read_csv(pth, index_col=0, )

        # File unlocking
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    hist.fillna(value="None", inplace=True)

    if row_number < 0:
        row_number += hist.last_valid_index() + 1

    single_row = hist.iloc[row_number]

    for entry_name in single_row.keys():
        if entry_name in config.parameters.keys() and entry_name not in config.readOnly:
            config(entry_name, single_row[entry_name])

    return config, row_number


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
