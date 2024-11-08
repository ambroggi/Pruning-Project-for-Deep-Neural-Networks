import os

import numpy as np
import pandas as pd
import plotly
import plotly.express
import plotly.graph_objects

readability = {
    "parameters": "Total number of parameters",
    "val_f1_score_macro": "F1 score in validation",
    "val_f1_score_weight": "F1 score (weighted by occurences)",
    "actual_parameters": "Number of non-zero parameters",
    "TimeForRun": "Time for normal fit after pruning",
    "TimeForPrune": "Time for pruning the model",
    "TimeForPruneAndRetrain": "Time for pruning the model"
}


colors = {
    "ADDM_Joint": "red",
    "BERT_theseus": "orange",
    "DAIS": "gold",
    "Normal_Run": "black",
    "RandomStructured": "limegreen",
    "Recreation_Run": "gray",
    "TOFD": "cyan",
    "iteritive_full_theseus": "blue",
    "thinet": "purple"
}


scatterpairs_scaled = [
    ("actual_parameters", "val_f1_score_macro"),
    ("actual_parameters", "val_f1_score_weight"),
    ("parameters", "val_f1_score_macro"),
    ("actual_parameters", "TimeForRun"),
    ("actual_parameters", "TimeForPruneAndRetrain"),
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
    df["PruningSelection"] = df["PruningSelection"].apply(lambda x: str.replace(str.replace(str.replace(x, "BERT_theseus_training|", ""), "iteritive_full_theseus_training|", ""), "DAIS_training|", ""))

    # Calculate the actual parameters out of the normal parameters
    df["actual_parameters"] = df["parameters"] - df["NumberOfZeros"]

    # Normalize time
    df["TimeForRun"] = df["TimeForRun"]/max(df["TimeForRun"])
    df["TimeForPrune"] = df["TimeForPrune"]/max(df["TimeForPrune"])
    df["TimeForPruneAndRetrain"] = df["TimeForPruneAndRetrain"]/max(df["TimeForPruneAndRetrain"])

    # Get dataframe for values before pruning
    row_numbers = df["AssociatedOriginalRow"].fillna({x: x for x in df["AssociatedOriginalRow"].index.values}).astype(int)
    test_for_missing = [x == y for x, y in enumerate(df.index.values)]
    assert False not in test_for_missing
    df_pre = df.iloc[row_numbers]

    # Create scaled version of dataframe based on the pretrained values
    df_scaled = df.copy()
    numerical = ["val_f1_score_macro", "val_f1_score_weight", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune", "TimeForPruneAndRetrain"]
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

    pt = df.pivot_table(values=["val_f1_score_macro", "val_f1_score_weight", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune", "Memory", "CudaMemory", "GarbageCollectionSizeTotal", "TimeForPruneAndRetrain"], index=["PruningSelection", "WeightPrunePercent"], columns=[], aggfunc=["mean", lambda x: np.std(x)/(len(x)**0.5)])
    pt_scaled = df_scaled.pivot_table(values=["val_f1_score_macro", "val_f1_score_weight", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune", "TimeForPruneAndRetrain"], index=["PruningSelection", "WeightPrunePercent"], columns=[], aggfunc=["mean", lambda x: np.std(x)/(len(x)**0.5)])
    print(df.head())
    print(pt)
    return df, pt, pt_scaled


def graph_pt(pt: pd.DataFrame, pair: tuple[str, str] = ("actual_parameters", "val_f1_score_macro"), file: None | os.PathLike = None):
    plot = plotly.graph_objects.Figure()
    x_name, y_name = pair
    pt_err = pt["<lambda>"].T
    pt = pt["mean"].T

    sizes = {x: 2*(i) for i, x in enumerate({weight_prune_group for _, weight_prune_group in pt.keys()})}

    seen_keys = set()
    for pruning_selection, weight_prune_group in pt.keys():
        if pruning_selection not in seen_keys:
            seen_keys.add(pruning_selection)
            size = sizes[weight_prune_group] * 0 + 10
            # print(pt[pruning_selection])
            plot.add_trace(plotly.graph_objects.Scatter(x=pt[pruning_selection].T[x_name], y=pt[pruning_selection].T[y_name],
                                                        name=pruning_selection, marker={"size": size, "color": colors.get(pruning_selection, "Gray")}, text=pt[pruning_selection].keys(),
                                                        error_x=dict(type="data", array=list(pt_err[pruning_selection].T[x_name]), visible=True, color="rgba(0, 0, 0, 0.1)"),
                                                        error_y=dict(type="data", array=list(pt_err[pruning_selection].T[y_name]), visible=True, color="rgba(0, 0, 0, 0.1)")))  # ref: https://plotly.com/python/error-bars/

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

    if "f1" in pair[1]:
        plot.update({"layout": {"yaxis": {"range": [-0.1, None]}}})

    if "Time" in y_name or y_name in []:
        plot.update({"layout": {"yaxis": {"type": "log"}}})
    # "yaxis": {"range": [-0.1, 1.1]}},
    plot.update_layout(title_text=f"{readability.get(y_name, y_name)} vs {readability.get(x_name, x_name)}", title_xanchor="right")

    if file is not None:
        plot.update({"layout": {"title": {"xanchor": "right", "x": 0.9}}})
        plot.write_image(file, width=700, height=500, scale=2)
    else:
        plot.show()


if __name__ == "__main__":
    df_small, pt_small, pt_scaled_small = read_results("results/Small(v0.124).csv")
    df, pt, pt_scaled = read_results("results/Bigcomputer(v0.124).csv")

    combined = pd.concat([pt_small["mean"][["actual_parameters", "val_f1_score_macro"]], pt["mean"][["actual_parameters", "val_f1_score_macro"]]], axis=1, keys=["small", "big"]).astype(str)
    combined.sort_index(ascending=False, inplace=True)
    combined.loc[:, (["small", "big"], "actual_parameters")] = combined.loc[:, (["small", "big"], "actual_parameters")].map(lambda x: "N/A" if pd.isna(float(x)) else (str(int(float(x)//1000))+"k"))
    combined.rename(index=lambda x: str.replace(x, "_", " ") if "[" not in x else x[1:5], inplace=True)
    combined.rename(columns=readability, inplace=True)
    combined.to_latex("results/images/table.txt", float_format="%.5f", longtable=True)

    if not False:  # Just for fun, every time I disable this I am just going to add another "not" here
        for x in scatterpairs_scaled:
            graph_pt(pt_scaled, x, file=f"results/images/Big-scaled-{x[0]}-{x[1]}.png")
            graph_pt(pt_scaled_small, x, file=f"results/images/Small-scaled-{x[0]}-{x[1]}.png")
            # graph_pt(pt, x)
        for x in scatterpairs_true:
            graph_pt(pt, x, file=f"results/images/Big-{x[0]}-{x[1]}.png")
            graph_pt(pt_small, x, file=f"results/images/Small-{x[0]}-{x[1]}.png")
