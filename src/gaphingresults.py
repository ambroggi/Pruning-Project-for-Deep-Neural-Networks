import pandas as pd
import numpy as np
import os
import plotly
import plotly.express
import plotly.graph_objects


readability = {
    "parameters": "Total number of parameters",
    "val_f1_score_macro": "F1 score in validation",
    "val_f1_score_weight": "F1 score (weighted by occurences)",
    "actual_parameters": "Number of non-zero parameters",
    "TimeForRun": "Time for normal fit after pruning",
    "TimeForPrune": "Time for pruning the model"
}


scatterpairs_scaled = [
    ("actual_parameters", "val_f1_score_macro"),
    ("actual_parameters", "val_f1_score_weight"),
    ("parameters", "val_f1_score_macro"),
    ("actual_parameters", "TimeForRun"),
    ("actual_parameters", "TimeForPrune"),
    ("val_f1_score_weight", "val_f1_score_macro")
]

scatterpairs_true = [
    ("actual_parameters", "val_f1_score_macro"),
    ("actual_parameters", "Memory"),
    ("actual_parameters", "CudaMemory"),
    ("actual_parameters", "GarbageCollectionSizeTotal")
]


def read_results(path: str | os.PathLike = "results/record.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, index_col=0)

    # Strip the true version number out of the version id
    df["Version"] = df["Version"].apply(lambda x: int(str(x).split(" - ")[-1])).astype(int)

    # Replace NANs with Normal Run
    df["PruningSelection"] = df["PruningSelection"].fillna("Normal_Run")

    # Calculate the actual parameters out of the normal parameters
    df["actual_parameters"] = df["parameters"] - df["NumberOfZeros"]

    # Normalize time
    df["TimeForRun"] = df["TimeForRun"]/max(df["TimeForRun"])
    df["TimeForPrune"] = df["TimeForPrune"]/max(df["TimeForPrune"])

    # Get dataframe for values before pruning
    row_numbers = df["AssociatedOriginalRow"].fillna({x: x for x in df["AssociatedOriginalRow"].index.values}).astype(int)
    test_for_missing = [x == y for x, y in enumerate(df.index.values)]
    assert False not in test_for_missing
    df_pre = df.iloc[row_numbers]

    # Create scaled version of dataframe based on the pretrained values
    df_scaled = df.copy()
    numerical = ["val_f1_score_macro", "val_f1_score_weight", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune"]
    for x in numerical:
        df_scaled[x] = df[x].values/df_pre[x].values
        df_scaled[x].fillna(-1)

    # Remove runs that did not finish.
    df = df[~df["parameters"].isna()]
    df.reset_index(inplace=True)

    # Get only the current version (Done this late just so the scaling is not indexing into a Null)
    # df = df[df["Version"] >= df["Version"].max()]
    # df = df[df["LengthOfTrainingData"] > 1000]
    df = df[df["Notes"] == 0]
    # df_scaled = df_scaled[df_scaled["LengthOfTrainingData"] > 1000]
    df_scaled = df_scaled[df_scaled["Notes"] == 0]

    pt = df.pivot_table(values=["val_f1_score_macro", "val_f1_score_weight", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune", "Memory", "CudaMemory", "GarbageCollectionSizeTotal"], index=["PruningSelection", "WeightPrunePercent"], columns=[], aggfunc=["mean", lambda x: np.std(x)/(len(x)**0.5)])
    pt_scaled = df_scaled.pivot_table(values=["val_f1_score_macro", "val_f1_score_weight", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune"], index=["PruningSelection", "WeightPrunePercent"], columns=[], aggfunc=["mean", lambda x: np.std(x)/(len(x)**0.5)])
    print(df.head())
    print(pt)
    return df, pt, pt_scaled


def graph_pt(pt: pd.DataFrame, pair: tuple[str, str] = ("actual_parameters", "val_f1_score_macro")):
    plot = plotly.graph_objects.Figure()
    x_name, y_name = pair
    pt_err = pt["<lambda>"].T
    pt = pt["mean"].T

    sizes = {x: 2*(i) for i, x in enumerate({weight_prune_group for _, weight_prune_group in pt.keys()})}

    seen_keys = set()
    for pruning_selection, weight_prune_group in pt.keys():
        if pruning_selection not in seen_keys:
            seen_keys.add(pruning_selection)
            size = sizes[weight_prune_group] * 0 + 20
            # print(pt[pruning_selection])
            plot.add_trace(plotly.graph_objects.Scatter(x=pt[pruning_selection].T[x_name], y=pt[pruning_selection].T[y_name],
                                                        name=pruning_selection, marker={"size": size}, text=pt[pruning_selection].keys(),
                                                        error_x=dict(type="data", array=list(pt_err[pruning_selection].T[x_name]), visible=True),
                                                        error_y=dict(type="data", array=list(pt_err[pruning_selection].T[y_name]), visible=True)))  # ref: https://plotly.com/python/error-bars/

    plot.update({
        "layout": {"title": {
                        "text": f"{readability.get(y_name, y_name)} vs {readability.get(x_name, x_name)}",
                        "xanchor": "center",
                        'x': 0.5},
                   "xaxis_title": f"{readability.get(x_name, x_name)}",
                   "yaxis_title": f"{readability.get(y_name, y_name)}",
                   "legend_title": "Method of Pruning",
                   "xaxis": {"type": "linear", "autorange": "reversed"}}
        })

    if "Time" in y_name or y_name in []:
        plot.update({"layout": {"yaxis": {"type": "log"}}})
    # "yaxis": {"range": [-0.1, 1.1]}},
    plot.update_layout(title_text=f"{readability.get(y_name, y_name)} vs {readability.get(x_name, x_name)}", title_xanchor="right")
    plot.show()


if __name__ == "__main__":
    df_small, pt_small, pt_scaled_small = read_results("results/Smallcomputer(v0.96).csv")
    df, pt, pt_scaled = read_results("results/Bigcomputer(v0.99).csv")
    # df, pt, pt_scaled = read_results("results/record.csv")
    for x in scatterpairs_scaled:
        graph_pt(pt_scaled, x)
        graph_pt(pt_scaled_small, x)
        # graph_pt(pt, x)
    for x in scatterpairs_true:
        graph_pt(pt, x)
        graph_pt(pt_small, x)
