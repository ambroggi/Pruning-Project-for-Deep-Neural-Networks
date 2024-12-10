import os

import numpy as np
import pandas as pd
import plotly
import plotly.express
import plotly.graph_objects

readability = {
    "parameters": "Total number of parameters",
    "val_f1_score_macro": "F1 score in validation",
    "val_f1_score_weight": "F1 score (weighted by occurrences)",
    "actual_parameters": "Number of non-zero parameters",
    "TimeForRun": "Time for normal fit after pruning",
    "TimeForPrune": "Time for pruning the model",
    "TimeForPruneAndRetrain": "Time for pruning the model",
    "ADDM_Joint": "ADMM Joint",
    "ADMM_Joint": "ADMM Joint",
    "admm_Joint": "ADMM Joint",
    "BERT_theseus": "BERT Theseus",
    "DAIS": "DAIS",
    "Normal_Run": "Original Run",
    "RandomStructured": "Random Structured",
    "Recreation_Run": "Recreation",
    "TOFD": "TOFD",
    "iterative_full_theseus": "Iterative Theseus",
    "iteritive_full_theseus": "Iterative Theseus",
    "thinet": "Thinet",
    "F1 score in validation": "F1 Score",
    "Time for pruning the model": "Pruning Time, Scaled",
    "Number of non-zero parameters": "Parameters",
    "PruningSelection": "Algorithm"
}

extra_readability = {
    "BERT_theseus_training|": "",
    "iterative_full_theseus_training|": "",
    "iteritive_full_theseus_training|": "",
    "DAIS_training|": ""
}

colors = {
    "ADMM Joint": "red",
    "BERT Theseus": "orange",
    "DAIS": "gold",
    "Original Run": "black",
    "Random Structured": "limegreen",
    "Recreation": "gray",
    "TOFD": "cyan",
    "Iterative Theseus": "blue",
    "Thinet": "purple"
}


shapes = {
    "ADMM Joint": "x",
    "BERT Theseus": "cross",
    "DAIS": "square",
    "Original Run": "circle",
    "Random Structured": "triangle-up",
    "Recreation": "circle",
    "TOFD": "bowtie",
    "Iterative Theseus": "diamond",
    "Thinet": "star"
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


def original_run_top_sort_func(x):
    # This is a function used in a .map expression so that "Original Run" is sorted to the top, for visualization.
    # I just noticed that I was using this as a lambda a lot so just made it into a function
    return x if x != "Original Run" else "AAAA"


def error_bar(x: list[float]) -> float:
    # 1.960 is for 95% confidence interval
    return 1.960*np.std(x)/(len(x)**0.5)


def read_results(path: str | os.PathLike = "results/record.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, index_col=0)

    # Strip the true version number out of the version id
    df["Version"] = df["Version"].apply(lambda x: int(str(x).split(" - ")[-1])).astype(int)

    # Replace NANs with Normal Run
    df["PruningSelection"] = df["PruningSelection"].fillna("Normal_Run")
    for extra_str in extra_readability:
        df["PruningSelection"] = df["PruningSelection"].apply(lambda x: str.replace(x, extra_str, extra_readability[extra_str]))
    df["PruningSelection"] = df["PruningSelection"].map(readability)

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
    df = df[(df["Notes"] == 0) | (df["Notes"] == 8) | (df["Notes"] == 32)]
    df = df[df["PruningSelection"] != "TOFD"]  # Removing support for TOFD since it does not seem to be working despite efforts
    # df_scaled = df_scaled[df_scaled["LengthOfTrainingData"] > 1000]
    df_scaled = df_scaled[(df_scaled["Notes"] == 0) | (df["Notes"] == 8) | (df["Notes"] == 32)]
    df_scaled = df_scaled[df_scaled["PruningSelection"] != "TOFD"]  # Removing support for TOFD since it does not seem to be working despite efforts

    pt = df.pivot_table(values=["val_f1_score_macro", "val_f1_score_weight", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune", "Memory", "CudaMemory", "GarbageCollectionSizeTotal", "TimeForPruneAndRetrain"], index=["PruningSelection", "WeightPrunePercent"], columns=[], aggfunc=["mean", error_bar, "std"]).sort_index(level=0, sort_remaining=False, key=lambda x: x.map(original_run_top_sort_func))
    pt_scaled = df_scaled.pivot_table(values=["val_f1_score_macro", "val_f1_score_weight", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune", "TimeForPruneAndRetrain"], index=["PruningSelection", "WeightPrunePercent"], columns=[], aggfunc=["mean", error_bar, "std"]).sort_index(level=0, sort_remaining=False, key=lambda x: x.map(original_run_top_sort_func))
    # print(df.head())
    df.pivot_table(values=["val_f1_score_macro", "val_f1_score_weight", "parameters", "actual_parameters", "TimeForRun", "TimeForPrune", "Memory", "CudaMemory", "GarbageCollectionSizeTotal", "TimeForPruneAndRetrain"], index=["PruningSelection", "WeightPrunePercent"], columns=[], aggfunc="count").sort_index(level=0, sort_remaining=False, key=lambda x: x.map(original_run_top_sort_func)).to_csv(f"results/images/debug_{os.path.basename(path)}")
    return df, pt, pt_scaled


def graph_pt(pt: pd.DataFrame, XYpair: tuple[str, str] = ("actual_parameters", "val_f1_score_macro"), file: None | os.PathLike = None):
    plot = plotly.graph_objects.Figure()
    x_name, y_name = XYpair
    pt_err = pt["error_bar"].T
    pt = pt["mean"].T

    sizes = {x: 2*(i) for i, x in enumerate({weight_prune_group for _, weight_prune_group in pt.keys()})}

    seen_keys = set()
    for pruning_selection, weight_prune_group in pt.keys():
        if pruning_selection not in seen_keys:
            seen_keys.add(pruning_selection)
            size = sizes[weight_prune_group] * 0 + 10
            # print(pt[pruning_selection])
            plot.add_trace(plotly.graph_objects.Scatter(x=pt[pruning_selection].T[x_name], y=pt[pruning_selection].T[y_name],
                                                        name=pruning_selection, marker={"size": size, "color": colors.get(pruning_selection, "Gray"), "symbol": shapes.get(pruning_selection, "circle-open")}, text=pt[pruning_selection].keys(),
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

    if "f1" in XYpair[1]:
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


def make_table(pt1, pt2):
    combined = pd.concat([pt1["mean"][["actual_parameters", "val_f1_score_macro"]],
                          pt1["std"]["val_f1_score_macro"].rename("Standard Deviation"),
                          pt1["mean"][["TimeForPruneAndRetrain"]],
                          pt2["mean"][["actual_parameters", "val_f1_score_macro"]],
                          pt2["std"]["val_f1_score_macro"].rename("Standard Deviation"),
                          pt2["mean"][["TimeForPruneAndRetrain"]],
                          ], axis=1, keys=["small", "small", "small", "big", "big", "big"]).astype(object)

    combined.sort_index(ascending=False, inplace=True, level=1)
    combined.loc[:, (["small", "big"], "actual_parameters")] = combined.loc[:, (["small", "big"], "actual_parameters")].map(lambda x: pd.NA if pd.isna(float(x)) else (str(int(float(x)//1000))+"k"))
    combined.rename(index=lambda x: str.replace(x, "_", " ") if "[" not in x else x[1:5], inplace=True)
    combined.rename(columns=readability, inplace=True)
    combined.rename(columns=readability, inplace=True)
    combined = combined.convert_dtypes()
    # All this is just to get the algorithms sorted alphabetically but with the Original Run first (Basically sorts it assuming 'Original Run' means 'AAAA')
    combined.sort_index(inplace=True, level=0, sort_remaining=False, key=lambda x: x.map(original_run_top_sort_func))

    # https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
    styler = combined.style.highlight_max([("small", "F1 Score"), ("big", "F1 Score")], props='bfseries: ;')
    styler.format(subset=[("small", "F1 Score"), ("big", "F1 Score")], precision=3, na_rep="¬")
    styler.format(subset=[("small", "Pruning Time, Scaled"), ("big", "Pruning Time, Scaled")], precision=3, na_rep="¬")
    styler.format("±{:0.2f}", subset=[("small", "Standard Deviation"), ("big", "Standard Deviation")], precision=2)
    # https://stackoverflow.com/a/57152529
    # styler.background_gradient(cmap=matplotlib.colormaps['Greys'], axis=1)

    styler.to_latex("results/images/table.txt", environment="longtable", clines="skip-last;data", hrules=True)  # , longtable=True


if __name__ == "__main__":
    df_small, pt_small, pt_scaled_small = read_results("results/SmallModel(v0.131).csv")
    df, pt, pt_scaled = read_results("results/BigModel(v0.131).csv")

    make_table(pt_small, pt)

    if not False:  # Just for fun, every time I disable this I am just going to add another "not" here
        for x in scatterpairs_scaled:
            graph_pt(pt_scaled, x, file=f"results/images/Big-scaled-{x[0]}-{x[1]}.png")
            graph_pt(pt_scaled_small, x, file=f"results/images/Small-scaled-{x[0]}-{x[1]}.png")
            # graph_pt(pt, x)
        for x in scatterpairs_true:
            graph_pt(pt, x, file=f"results/images/Big-{x[0]}-{x[1]}.png")
            graph_pt(pt_small, x, file=f"results/images/Small-{x[0]}-{x[1]}.png")
