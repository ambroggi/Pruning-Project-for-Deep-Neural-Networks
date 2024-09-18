import pandas as pd
import os
import plotly
import plotly.express
import plotly.graph_objects


readability = {
    "parameters": "Total number of parameters",
    "val_f1_score_macro": "F1 score in validation",
    "actual_parameters": "Number of non-zero parameters",
    "TimeForRun": "Time for normal fit after pruning",
    "TimeForPrune": "Time for pruning the model"
}


scatterpairs = [
    ("parameters", "val_f1_score_macro"),
    ("actual_parameters", "val_f1_score_macro"),
    ("actual_parameters", "TimeForRun"),
    ("actual_parameters", "TimeForPrune")
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
    numerical = ["val_f1_score_macro", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune"]
    for x in numerical:
        df_scaled[x] = df[x].values/df_pre[x].values
        df_scaled[x].fillna(-1)

    # Remove runs that did not finish.
    df = df[~df["parameters"].isna()]
    df.reset_index(inplace=True)

    # Get only the current version (Done this late just so the scaling is not indexing into a Null)
    # df = df[df["Version"] >= df["Version"].max()]
    df = df[df["LengthOfTrainingData"] > 100000]

    pt = df.pivot_table(values=["val_f1_score_macro", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune"], index=["PruningSelection", "WeightPrunePercent"], columns=[], aggfunc="mean")
    pt_scaled = df_scaled.pivot_table(values=["val_f1_score_macro", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune"], index=["PruningSelection", "WeightPrunePercent"], columns=[], aggfunc="mean")
    print(df.head())
    print(pt)
    return df, pt, pt_scaled


def graph_pt(pt: pd.DataFrame, pair: int = 0):
    plot = plotly.graph_objects.Figure()
    x_name, y_name = scatterpairs[pair]
    pt = pt.T

    sizes = {x: 2*(i) for i, x in enumerate({weight_prune_group for _, weight_prune_group in pt.keys()})}

    seen_keys = set()
    for pruning_selection, weight_prune_group in pt.keys():
        if pruning_selection not in seen_keys:
            seen_keys.add(pruning_selection)
            size = sizes[weight_prune_group] * 0 + 20
            # print(pt[pruning_selection])
            plot.add_trace(plotly.graph_objects.Scatter(x=pt[pruning_selection].T[x_name], y=pt[pruning_selection].T[y_name], name=pruning_selection, marker={"size": size}, text=pt[pruning_selection].keys()))

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

    if "Time" in y_name:
        plot.update({"layout": {"yaxis": {"type": "log"}}})
    # "yaxis": {"range": [-0.1, 1.1]}},
    plot.update_layout(title_text=f"{readability.get(y_name, y_name)} vs {readability.get(x_name, x_name)}", title_xanchor="right")
    plot.show()


if __name__ == "__main__":
    df, pt, pt_scaled = read_results("results/Bigcomputer(v0.89)preview.csv")
    for x in range(len(scatterpairs)):
        graph_pt(pt_scaled, x)
