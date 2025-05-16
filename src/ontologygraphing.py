# This file has kind of been Frankensteined together as we keep adding more ontological based graphs and they just get added here.
# It creates outputs in a couple of different places, mainly in results/images
# And it reads in from the csv files kept in results.
import os
import sys
from itertools import repeat, product
from typing import Literal

# import numpy as np
import pandas as pd
import numpy as np
import plotly
import plotly.express
import plotly.graph_objects
import plotly.subplots
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
        plot.add_trace(plotly.graph_objects.Scatter(x=vals.index, y=vals,
                                                    name=pruning_selection,
                                                    marker={"color": colors.get(pruning_selection, "Gray"),
                                                            "symbol": shapes.get(pruning_selection, "circle-open")},
                                                    text=pt[pruning_selection].keys()))

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


def generate_table(dataframe: pd.DataFrame, file_name: str, col_name: str = "Number of connected", grouping: str = "Original Run", reduce: Literal["Horizontal", "Vertical"] = ""):
    print(f"Generating LaTeX table {file_name}")
    # https://stackoverflow.com/a/74025617
    only_original = dataframe.loc[dataframe["pruning type"] == grouping].copy()
    only_original.loc[:, "layer"] = r"\rotatebox{90}{Layer}"
    only_original.loc[:, "col_name"] = col_name
    pt = only_original.assign(vals=1).pivot_table(values="vals", columns=["col_name", col_name], index=["layer", "Layer"], aggfunc="count", fill_value=0)
    pt.index.set_names([None, None], inplace=True)
    pt.columns.set_names([None, None], inplace=True)
    # https://stackoverflow.com/a/63896673
    exterior_column = repeat(pt.columns[0][0])
    interior_columns = range(1, 11)
    cols = pt.columns.union([*zip(exterior_column, interior_columns)], sort=True)
    print(cols)
    pt = pt.reindex(cols, axis=1, fill_value=0)

    if "Horizontal" in reduce:
        grouping_size = 2
        # # https://www.reddit.com/r/learnpython/comments/nkvusr/how_can_i_group_and_sum_certain_columns_of_a/
        # bins = pd.cut(interior_columns, range(1, 11, grouping_size), include_lowest=True, right=False)
        # pt = pt.rename(columns=dict(zip(pt.columns, bins)))
        # pt = pt.groupby(pt.columns, axis=1).sum()
        interior_columns = [f"{x}-{min(x+grouping_size-1,max(interior_columns))}" for x in range(1, 11, grouping_size)]
        new_cols = [*zip(exterior_column, interior_columns)]
        for (x, y) in new_cols:
            y_start = int(y.split("-")[0])
            y_end = int(y.split("-")[1])+1
            aggregate = pt[[*zip(exterior_column, range(y_start, y_end))]].sum(axis=1)
            pt[x, y] = aggregate
        pt = pt.reindex(new_cols, axis=1)
        print(pt)
    if "Vertical" in reduce:
        grouping_size = 3
        row_names = [f"{x}-{min(x+grouping_size-1,len(pt)-1)}" for x in range(0, len(pt), grouping_size)]
        row_exterior = repeat(pt.index[0][0])
        for y in row_names:
            y_start = int(y.split("-")[0])
            y_end = int(y.split("-")[1])+1
            aggregate = pt.loc[[*zip(row_exterior, range(y_start, y_end))]].sum(axis=0)
            pt.loc[(next(row_exterior), y), :] = aggregate

        pt = pt.reindex([*zip(row_exterior, row_names)], axis=0)
        print(pt)

    # print(pt)
    st = pt.style
    st.background_gradient(cmap="inferno", vmin=0, vmax=max(pt.max()))
    st.to_latex(f"results/images/{file_name}.txt", convert_css=True, hrules=True)


def check_statistical_for_top_down():
    print("Generating P-value table, original run vs 500 random")
    sample_total = format_df("top_down_connections")
    sample_original = sample_total.loc[sample_total["pruning type"] == "Original Run"].copy()
    pt_sample = sample_original.assign(vals=1).pivot_table(values="vals", columns="Number of connected", index=["Layer", "csv row"], aggfunc="count", fill_value=0)
    cols = pt_sample.columns.union(range(1, 11), sort=True)
    pt_sample = pt_sample.reindex(cols, axis=1, fill_value=0)

    distribution = format_df("500-Random/top_down_connections")
    pt_dist = distribution.assign(vals=1).pivot_table(values="vals", columns="Number of connected", index=["Layer", "csv row"], aggfunc="count", fill_value=0)
    cols = pt_dist.columns.union(range(1, 11), sort=True)
    pt_dist = pt_dist.reindex(cols, axis=1, fill_value=0)

    # Create a blank table and then fill the table with the correct values
    pt_pvalue = pt_sample.loc[(slice(None), 0), (slice(None))].copy().astype("float")
    pt_pvalue.index = pt_pvalue.index.get_level_values(0)  # https://stackoverflow.com/a/51537177
    pt_statistic = pt_sample.loc[(slice(None), 0), (slice(None))].copy().astype("float")
    pt_statistic.index = pt_pvalue.index.get_level_values(0)
    for pair in product(pt_pvalue.index, pt_pvalue.columns):
        ks_2 = stats.ks_2samp(pt_dist.loc[(pair[0], slice(None)), (pair[1])], pt_sample.loc[(pair[0], slice(None)), (pair[1])])
        pval = ks_2.pvalue.item()
        statistic = ks_2.statistic.item()
        # print(pval)
        pt_pvalue.loc[pair] = pval
        pt_statistic.loc[pair] = statistic
    # print(pt_pvalue)
    pt_pvalue.columns.set_names(["P-Values"], inplace=True)

    st = pt_pvalue.style
    st.background_gradient(cmap="magma_r", vmin=0, vmax=1)
    st.format(precision=2)
    st.to_latex("results/images/top_down_pvalues.txt", convert_css=True, hrules=True)

    generate_table(distribution, "top_down_table_rand_500", "Number of connected", "Random Ontology")
    # https://stackoverflow.com/a/50209193
    # print(stats.ks_2samp(pt_dist[1], pt_sample[1]))


def check_all_statistical_for_top_down(distribution_file="500-Random/top_down_connections", sample_file="top_down_connections", filtering=lambda x: x["pruning type"] == "RandomConnections"):
    distribution = format_df(distribution_file)
    distribution: pd.DataFrame = distribution[filtering(distribution)]
    pt_dist = distribution.assign(vals=1).pivot_table(values="vals", columns="Number of connected", index=["Layer", "csv row"], aggfunc="count", fill_value=0)
    cols = pt_dist.columns.union(range(1, 11), sort=True)
    pt_dist = pt_dist.reindex(cols, axis=1, fill_value=0)

    sample_total: pd.DataFrame = format_df(sample_file)
    np_arr = []
    names = sample_total["pruning type"].unique()
    for i, type_value in enumerate(names):
        sample_original = sample_total.loc[sample_total["pruning type"] == type_value].copy()
        pt_sample = sample_original.assign(vals=1).pivot_table(values="vals", columns="Number of connected", index=["Layer", "csv row"], aggfunc="count", fill_value=0)
        cols = pt_sample.columns.union(range(1, 11), sort=True)
        pt_sample = pt_sample.reindex(cols, axis=1, fill_value=0)

        # Create a blank table and then fill the table with the correct values
        pt_pvalue: pd.DataFrame = pt_sample.loc[(slice(None), pt_sample.index[0][1]), (slice(None))].copy().astype("float")
        pt_pvalue.index = pt_pvalue.index.get_level_values(0)  # https://stackoverflow.com/a/51537177

        pt_statistic = pt_pvalue.copy()
        for pair in product(pt_pvalue.index, pt_pvalue.columns):
            distribution_of_cell = pt_dist.loc[(pair[0], slice(None)), (pair[1])]
            sample_of_cell = pt_sample.loc[(pair[0], slice(None)), (pair[1])]
            ks_2 = stats.ks_2samp(distribution_of_cell, sample_of_cell)
            pval = ks_2.pvalue.item()
            statistic = ks_2.statistic.item()

            # print(pval)
            pt_pvalue.loc[pair] = 2*(1-pval)*ks_2.statistic_sign.item() if (pval < 0.005) else (1-pval)*ks_2.statistic_sign.item()
            pt_statistic.loc[pair] = statistic
        # print(pt_pvalue)
        pt_pvalue.columns.set_names(["P-Values"], inplace=True)
        pt_statistic.columns.set_names(["Statistic-Values"], inplace=True)

        # Make sure that all versions have the same length
        pt_pvalue = pt_pvalue.reindex(pt_dist.index.get_level_values(0).unique(), fill_value=0)
        # print(f"{type_value}, {len(pt_pvalue.to_numpy())=}")
        np_arr.append(pt_pvalue.to_numpy())

    np_arr = np.array(np_arr)

    plot = plotly.express.imshow(np_arr, facet_col=0, color_continuous_scale='RdBu_r', facet_col_wrap=min(7, len(np_arr)), range_color=[-2, 2])
    if "waypoint" in sample_file:
        # https://community.plotly.com/t/changing-label-of-plotly-express-facet-categories/28066/5
        plot.for_each_annotation(lambda x: x.update(text=f"Epoch {4*int(x.text.split('facet_col=')[1])}"))
    else:
        plot.for_each_annotation(lambda x: x.update(text=f"{names[int(x.text.split('facet_col=')[1])]}"))

    # file_name = "results/images/"+distribution_file.split("/")[-1]+"-".join(distribution["pruning type"].unique())+".png"
    # plot.write_image(file_name, width=1000, height=800, scale=2)
    plot.show()


def format_df(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(f"results/{file_name}.csv", header=1, sep=",", engine='python')
    df.loc[:, "pruning type"] = df["pruning type"].fillna("Normal_Run")
    df.loc[:, "csv row"] = df["csv row"].astype(str)
    df = df[df["csv row"].apply(str.isnumeric)]
    df.loc[:, "csv row"] = df["csv row"].astype(int)
    df.loc[:, "Layer"] = df["Layer"].astype(int)

    for extra_str in extra_readability:
        df.loc[:, "pruning type"] = df["pruning type"].apply(lambda x: str.replace(x, extra_str, extra_readability[extra_str]))
    df.loc[:, "pruning type"] = df["pruning type"].map(lambda x: readability[x] if x in readability.keys() else x)

    for x in [y for y in df.columns if "Number" in y]:
        df.loc[:, x] = df[x].astype(int)

    if "high_nodes_" in file_name:
        df["Number for calculation"] = df["Number of connected classes"]/df["Number of classes total"]

    return df


if __name__ == "__main__":
    # check_all_statistical_for_top_down()
    if "-h" in sys.argv:
        print("""
              This is the graphing function for the created ontological outputs generated from the file buildontology.py
              This function did not have enough use to create a proper command line argument page so this is not in the proper format.

              ontologygraphing.py
              \t- Runs the standard ontology graph generation, to be put in the results/ folder.

              ontologygraphing.py waypointing
              \t- Runs a comparison over training epochs to examine how the paths change.

              ontologygraphing.py waypointing {folder}
              \t- Runs a comparison over training epochs within a specific file along the path pruning/{folder}/top_down_connections vs the main waypoint to examine pruning changes the paths.

              -h Displays this help text.

              """, flush=True)

    if len(sys.argv) == 1:
        print("Running standard ontology graph generation")

        for file_ in titles.keys():
            if not os.path.exists(f"results/{file_}.csv"):
                continue

            df = format_df(file_)

            for x in [y for y in df.columns if "Number" in y]:
                print(f"creating file: {file_} - column name {x}")
                pt = df.pivot_table(values=x, **options[file_], fill_value=0)
                graph_pt(pt, file=file_)

            if file_ == "top_down_connections":
                generate_table(df, "top_down_table", "Number of connected", "Original Run")
                generate_table(df, "Random_Ontology_top_down_table", "Number of connected", "Random Ontology")
                generate_table(df, "top_down_table_thin", "Number of connected", "Original Run", reduce="Horizontal")
                generate_table(df, "Random_Ontology_top_down_table_thin", "Number of connected", "Random Ontology", reduce="Horizontal")

            if file_ == "high_nodes_along_connections":
                generate_table(df, "high_nodes_along_connections_table", "Number of connected classes", "Original Run")
                generate_table(df, "high_nodes_along_connections_table_short", "Number of connected classes", "Original Run", reduce="Vertical")
                generate_table(df, "random_high_nodes_along_connections_table", "Number of connected classes", "Random Ontology")

            if file_ == "high_nodes":
                generate_table(df, "high_nodes_total", "Number of meanings for node", "Original Run")
        check_statistical_for_top_down()
        print("Generating image for each pruning method. Note: Not saved, just appearing in browser")
        check_all_statistical_for_top_down("top_down_connections", "top_down_connections", lambda x: x["pruning type"] == "Original Run")
    elif (len(sys.argv) == 2) and (sys.argv[1] == "waypointing"):
        print("Running more comparisons based on waypoints taken of the model during training.")
        df = format_df("waypointing/top_down_connections")
        pt = df.assign(vals=1).pivot_table(values="vals", columns=["Number of connected", "pruning type"], index=["Layer"], aggfunc="count", fill_value=0)
        # print(*[f"{x}\n" for x in zip_longest(pt.columns, product(range(1, 11), [f"Waypoint_{x}" for x in range(15)]))])
        # print([*zip(*pt.columns.values)])
        pt = pt.reindex(product(range(1, 11), [f"Waypoint_{x}" for x in range(15)]), axis=1)
        pt = pt.apply(lambda x: x**0.5)
        pt.columns.names
        pt.index.names
        # plot = plotly.express.imshow(pt, title="Changes over training", x="Number of connected")

        np_test = np.array([pt.loc[:, (slice(None), x)].to_numpy() for x in [f"Waypoint_{x}" for x in range(13)]])
        np_test = np_test/np_test.max()
        # https://stackoverflow.com/a/66054748
        plot = plotly.express.imshow(np_test, facet_col=0, facet_col_wrap=7)
        plot.for_each_annotation(lambda x: x.update(text=f"Epoch {4*int(x.text.split('facet_col=')[1])}"))
        plot.write_image("results/images/top_down_connections_sqr.png", width=1000, height=800, scale=2)
        # plot.show()

        check_all_statistical_for_top_down("waypointing/top_down_connections", "waypointing/top_down_connections", filtering=lambda x: x["pruning type"] == "Waypoint_0")
        check_all_statistical_for_top_down("waypointing/top_down_connections", "waypointing/top_down_connections", filtering=lambda x: x["pruning type"] == "Waypoint_6")
        check_all_statistical_for_top_down("waypointing/top_down_connections", "waypointing/top_down_connections", filtering=lambda x: x["pruning type"] == "Waypoint_12")
    elif "waypointing":
        folder = sys.argv[2]  # "thinet"  # "randStructured"
        df = format_df(f"pruning/{folder}/top_down_connections")
        pt = df.assign(vals=1).pivot_table(values="vals", columns=["Number of connected", "pruning type"], index=["Layer"], aggfunc="count", fill_value=0)
        # print(*[f"{x}\n" for x in zip_longest(pt.columns, product(range(1, 11), [f"Waypoint_{x}" for x in range(15)]))])
        # print([*zip(*pt.columns.values)])
        percentage = [float(x.split("|")[1]) for x in df["pruning type"].unique()]
        percentage.sort(reverse=True)
        pt = pt.reindex(product(range(1, 11), [f"{folder}|{x}" for x in percentage]), axis=1, fill_value=0)
        pt = pt.apply(lambda x: x**0.5)
        pt.columns.names
        pt.index.names
        # plot = plotly.express.imshow(pt, title="Changes over training", x="Number of connected")

        np_test = np.array([pt.loc[:, (slice(None), x)].to_numpy() for x in [f"{folder}|{x}" for x in percentage]])
        np_test = np_test/np_test.max()
        # https://stackoverflow.com/a/66054748
        plot = plotly.express.imshow(np_test, facet_col=0)
        plot.for_each_annotation(lambda x: x.update(text=f"{percentage[int(x.text.split('facet_col=')[1])]}"))
        plot.show()

        check_all_statistical_for_top_down("waypointing/top_down_connections", f"pruning/{folder}/top_down_connections", lambda x: x["pruning type"] == "Waypoint_12")
