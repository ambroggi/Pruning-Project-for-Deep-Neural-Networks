import pandas as pd
import os
import plotly
import plotly.express
import plotly.graph_objects


def read_results(path: str | os.PathLike = "results/record.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, index_col=0)

    # Remove runs that did not finish.
    df = df[~df["parameters"].isna()]

    # Strip the true version number out of the version id
    df["Version"] = df["Version"].apply(lambda x: int(str(x).split(" - ")[-1])).astype(int)

    # Replace NANs with Normal Run
    df["PruningSelection"] = df["PruningSelection"].fillna("Normal_Run")

    # Calculate the actual parameters out of the normal parameters
    df["actual_parameters"] = df["parameters"] - df["NumberOfZeros"]
    pt = df.pivot_table(values=["val_f1_score", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune"], index=["PruningSelection", "ProcessID"], columns=[], aggfunc="mean")
    print(df.head())
    print(pt)
    return df, pt


def graph_pt(pt: pd.DataFrame):
    plot = plotly.graph_objects.Figure()

    for x in pt[:]:
        print(x)
        plot.add_trace(plotly.graph_objects.scatter(x, x="parameters", y="val_f1_score", log_x=True, labels="index", range_y=(0, 1)))
    plot.show()


if __name__ == "__main__":
    df, pt = read_results()
    graph_pt(pt)
