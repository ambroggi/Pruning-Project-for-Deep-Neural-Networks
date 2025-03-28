import os
from itertools import repeat, product

# import numpy as np
import pandas as pd
import plotly
import plotly.express
import plotly.graph_objects
from scipy import stats


readability = {
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
    "iteritive_full_theseus": "Iterative Theseus",  # Backwards compatibility with misspelled version.
    "thinet": "Thinet",
    "PruningSelection": "Algorithm",
    "RandomConnections": "Random Ontology"
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
    "Thinet": "purple",
    "Random Ontology": "darkslategray"
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
    "Thinet": "star",
    "Random Ontology": "square"
}


titles = {
    "top_down_connections": ("Top-down Connections", "Total Connection Paths to Final Layers", "reversed"),
    "high_nodes_along_connections": ("High Nodes Along Primary Path", "Average Connection Along Path", "normal"),
    "high_nodes_of_reduced_classes": ("High Nodes Along Primary Path", "Average Connection Along Path", "normal"),
    "high_nodes": ("High Nodes Total", "Concentration Average of High Nodes", "normal"),
    "bottom_up_connections": ("Bottom-Up Connections", "Total Connection Paths that Lead to First Layers", "normalog"),
}


options = {
    "top_down_connections": {"columns": ['pruning type'], "aggfunc": "sum", "index": ["Layer"]},
    "high_nodes_along_connections": {"columns": ['pruning type'], "aggfunc": (lambda x: (sum(x)/len(x))), "index": ["Layer"]},
    "high_nodes_of_reduced_classes": {"columns": ['pruning type'], "aggfunc": (lambda x: (sum(x)/len(x))), "index": ["Layer"]},
    "high_nodes": {"columns": ['pruning type'], "aggfunc": (lambda x: (sum(x)/len(x))), "index": ["Layer"]},
    "bottom_up_connections": {"columns": ['pruning type'], "aggfunc": (lambda x: (sum(x)/len(x))), "index": ["Layer"]}
}


def graph_pt(pt: pd.DataFrame, file: None | os.PathLike = None):
    plot = plotly.graph_objects.Figure()

    for pruning_selection in pt.keys():
        vals = pt[pruning_selection]
        plot.add_trace(plotly.graph_objects.Scatter(x=vals.index, y=vals, name=pruning_selection, marker={"color": colors.get(pruning_selection, "Gray"), "symbol": shapes.get(pruning_selection, "circle-open")}, text=pt[pruning_selection].keys()))

    x_axis_config = {"autorange": "reversed", "type": "linear"} if "reversed" in titles.get(file, ("Unknown", "unknown", "unknown"))[2] else {"type": "linear"}
    y_axis_config = {"type": "log"} if "log" in titles.get(file, ("Unknown", "unknown", "unknown"))[2] else {"type": "linear"}

    # Just noting that looking up all of the possible plotly codes is quite annoying. The documentation is difficult to parse.
    plot.update({
        "layout": {"title": {
                        "text": titles.get(file, ("Unknown", "unknown"))[0],
                        "xanchor": "left",
                        'x': 0.3,
                        "font": {"size": 13}},
                   "xaxis_title": "Layer",
                   "yaxis_title": titles.get(file, ("Unknown", "unknown"))[1],
                   "legend_title": "Method of Pruning",
                   "xaxis": x_axis_config,
                   "yaxis": y_axis_config},
        })

    # plot.update_layout(title_text=f"{readability.get(y_name, y_name)} vs {readability.get(x_name, x_name)}", title_xanchor="right")

    if file is not None:
        plot.update({"layout": {"title": {"xanchor": "right", "x": 0.9}}})
        plot.write_image(f"results/images/{file}.png", width=1400, height=500, scale=2)
    else:
        plot.show()


def generate_table(dataframe: pd.DataFrame, file_name: str, col_name: str = "Number of connected", grouping: str = "Original Run"):
    # https://stackoverflow.com/a/74025617
    only_original = dataframe.loc[dataframe["pruning type"] == grouping].copy()
    only_original.loc[:, "layer"] = r"\rotatebox{90}{Layer}"
    only_original.loc[:, "col_name"] = col_name
    pt = only_original.assign(vals=1).pivot_table(values="vals", columns=["col_name", col_name], index=["layer", "Layer"], aggfunc="count", fill_value=0)
    pt.index.set_names([None, None], inplace=True)
    pt.columns.set_names([None, None], inplace=True)
    # https://stackoverflow.com/a/63896673
    cols = pt.columns.union([*zip(repeat("Number of connected classes"), range(1, 11))], sort=True)
    print(cols)
    pt = pt.reindex(cols, axis=1, fill_value=0)
    # print(pt)
    st = pt.style
    st.background_gradient(cmap="inferno", vmin=0, vmax=max(pt.max()))
    st.to_latex(f"results/images/{file_name}.txt", convert_css=True, hrules=True)


def check_statistical_for_top_down():
    sample_total = format_df("top_down_connections")
    sample_original = sample_total.loc[sample_total["pruning type"] == "Original Run"].copy()
    pt_sample = sample_original.assign(vals=1).pivot_table(values="vals", columns="Number of connected", index=["Layer", "csv row"], aggfunc="count", fill_value=0)
    cols = pt_sample.columns.union(range(1, 11), sort=True)
    pt_sample = pt_sample.reindex(cols, axis=1, fill_value=0)

    distribution = format_df("500-Random/top_down_connections")
    pt_dist = distribution.assign(vals=1).pivot_table(values="vals", columns="Number of connected", index=["Layer", "csv row"], aggfunc="count", fill_value=0)
    cols = pt_dist.columns.union(range(1, 11), sort=True)
    pt_dist = pt_dist.reindex(cols, axis=1, fill_value=0)

    pt_pvalue = pt_sample.loc[(slice(None), 0), (slice(None))].copy().astype("float")
    pt_pvalue.index = pt_pvalue.index.get_level_values(0)  # https://stackoverflow.com/a/51537177
    for pair in product(pt_pvalue.index, pt_pvalue.columns):
        pval = stats.ks_2samp(pt_dist.loc[(pair[0], slice(None)), (pair[1])], pt_sample.loc[(pair[0], slice(None)), (pair[1])]).pvalue.item()
        # print(pval)
        pt_pvalue.loc[pair] = pval
    # print(pt_pvalue)
    pt_pvalue.columns.set_names(["P-Values"], inplace=True)

    st = pt_pvalue.style
    st.background_gradient(cmap="magma_r", vmin=0, vmax=1)
    st.format(precision=2)
    st.to_latex("results/images/top_down_pvalues.txt", convert_css=True, hrules=True)

    # https://stackoverflow.com/a/50209193
    # print(stats.ks_2samp(pt_dist[1], pt_sample[1]))


def format_df(file_name: str):
    df = pd.read_csv(f"results/{file_name}.csv", header=1, sep=",", engine='python')
    df.loc[:, "pruning type"] = df["pruning type"].fillna("Normal_Run")
    df.loc[:, "csv row"] = df["csv row"].astype(str)
    df = df[df["csv row"].apply(str.isnumeric)]
    df.loc[:, "csv row"] = df["csv row"].astype(int)
    df.loc[:, "Layer"] = df["Layer"].astype(int)

    for extra_str in extra_readability:
        df.loc[:, "pruning type"] = df["pruning type"].apply(lambda x: str.replace(x, extra_str, extra_readability[extra_str]))
    df.loc[:, "pruning type"] = df["pruning type"].map(readability)

    for x in [y for y in df.columns if "Number" in y]:
        df.loc[:, x] = df[x].astype(int)

    if "high_nodes_" in file_name:
        df["Number for calculation"] = df["Number of connected classes"]/df["Number of classes total"]

    return df


if __name__ == "__main__":
    check_statistical_for_top_down()
    for file_ in titles.keys():
        if not os.path.exists(f"results/{file_}.csv"):
            continue
        
        df = format_df(file_)
        
        for x in [y for y in df.columns if "Number" in y]:
            print(f"{file_} - {x}")
            pt = df.pivot_table(values=x, **options[file_], fill_value=0)

            graph_pt(pt, file=file_)

        if file_ == "top_down_connections":
            generate_table(df, "top_down_table", "Number of connected", "Original Run")
            generate_table(df, "Random_Ontology_top_down_table", "Number of connected", "Random Ontology")

        if file_ == "high_nodes_along_connections":
            generate_table(df, "high_nodes_along_connections_table", "Number of connected classes", "Original Run")
            generate_table(df, "random_high_nodes_along_connections_table", "Number of connected classes", "Random Ontology")

        if file_ == "high_nodes":
            generate_table(df, "high_nodes_total", "Number of meanings for node", "Original Run")
