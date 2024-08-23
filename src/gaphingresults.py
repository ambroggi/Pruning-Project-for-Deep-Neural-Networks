import pandas as pd
import os


def read_results(path: str | os.PathLike = "results/record.csv"):
    df = pd.read_csv(path, index_col=0)

    # Remove runs that did not finish.
    df = df[~df["parameters"].isna()]

    # Strip the true version number out of the version id
    df["Version"] = df["Version"].apply(lambda x: int(str(x).split(" - ")[-1])).astype(int)

    # Replace NANs with Normal Run
    df["PruningSelection"] = df["PruningSelection"].fillna("Normal_Run")

    # Calculate the actual parameters out of the normal parameters
    df["actual_parameters"] = df["parameters"] - df["NumberOfZeros"]
    pt = df.pivot_table(values=["val_f1_score", "parameters", "actual_parameters", "TimeForRun"], index="PruningSelection", columns=[], aggfunc="mean")
    print(df.head())
    print(pt)


if __name__ == "__main__":
    read_results()
