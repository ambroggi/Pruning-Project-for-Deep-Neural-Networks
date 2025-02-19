import os

# import numpy as np
import pandas as pd
import plotly
import plotly.express
import plotly.graph_objects


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
        plot.write_image(f"results/images/{file}.png", width=700, height=500, scale=2)
    else:
        plot.show()


if __name__ == "__main__":
    for file_ in titles.keys():
        if not os.path.exists(f"{file_}.csv"):
            continue
        df = pd.read_csv(f"{file_}.csv", header=1, sep=",", engine='python')
        df["pruning type"] = df["pruning type"].fillna("Normal_Run")
        df["csv row"] = df["csv row"].astype(str)
        df = df[df["csv row"].apply(str.isnumeric)]
        df["csv row"] = df["csv row"].astype(int)
        df["Layer"] = df["Layer"].astype(int)

        for extra_str in extra_readability:
            df["pruning type"] = df["pruning type"].apply(lambda x: str.replace(x, extra_str, extra_readability[extra_str]))
        df["pruning type"] = df["pruning type"].map(readability)

        for x in [y for y in df.columns if "Number" in y]:
            df[x] = df[x].astype(int)

        if "high_nodes_" in file_:
            df["Number for calculation"] = df["Number of connected classes"]/df["Number of classes total"]

        for x in [y for y in df.columns if "Number" in y]:
            pt = df.pivot_table(values=x, **options[file_])
            pt = pt.fillna(0)

            graph_pt(pt, file=file_)
