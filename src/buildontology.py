if __name__ == "__main__":
    import os
    from random import random

    import pandas as pd
    import rdflib
    import torch.nn
    from rdflib.namespace import RDF, RDFS
    from rdflib.plugins.sparql import prepareQuery

    import __init__ as src
    import extramodules

    NNC = rdflib.Namespace("https://github.com/ambroggi/Pruning-Project-for-Deep-Neural-Networks")   # Neural Network Connections (I guess I should have a location that is not example.org?)
    QINFO = prepareQuery("""
        SELECT ?csv_row_number ?csv_name ?pruning_type (COUNT(distinct ?class) as ?num_classes_total) (COUNT(distinct ?tag) as ?num_classes_reduced)
        WHERE {
            ?model ns1:filename ?csv_name .
            ?model ns1:prune_type ?pruning_type .
            ?model ns1:csv_row ?csv_row_number .
            ?class ns1:training_percent ?training .
            OPTIONAL {
                    FILTER(?training > 0.5)
                    ?class ns1:name ?tag .
                    }
        } GROUP BY ?csv_row_number ?csv_name ?pruning_type
        """, {"ns1": NNC, "rdfs": RDFS})
    Q1 = prepareQuery("""
        SELECT ?l_idx (SAMPLE(?input_meaning) as ?input) (COUNT(DISTINCT ?mean) as ?num_paths)
        WHERE {
            ?node ns1:layer ?layer .
            ?layer ns1:layer_index ?l_idx .
            ?node ns1:node_index ?n_idx .
            ?meaning a ns1:class_node .
            ?meaning ns1:associated_node/(ns1:secondary_contributor | ns1:primary_contributor)+ ?node .
            ?meaning ns1:name ?mean .
            OPTIONAL {
                ?node ns1:meaning ?in_meaning .
                ?in_meaning ns1:name ?input_meaning .
            }
        } GROUP BY ?n_idx ?l_idx
        ORDER BY desc(?l_idx) ?n_idx
        """, {"ns1": NNC, "rdfs": RDFS})

    Q2 = prepareQuery("""
        # Get "most important" input numbers counts
        SELECT ?l_idx (SAMPLE(?input_meaning) as ?input) (COUNT(DISTINCT ?mean) as ?num_paths)
        WHERE {
            ?node ns1:layer ?layer .
            ?layer ns1:layer_index ?l_idx .
            ?node ns1:node_index ?n_idx .
            ?meaning ns1:associated_node ?start .
            ?start a ns1:input_node .
            ?node (ns1:secondary_contributor | ns1:primary_contributor)+ ?start .
            ?meaning ns1:name ?mean .
            ?meaning ns1:mean_type ns1:by_definition .
            OPTIONAL {
                ?node ns1:meaning ?in_meaning .
                ?in_meaning ns1:name ?input_meaning .
            }
        } GROUP BY ?n_idx ?l_idx
        ORDER BY desc(?l_idx) ?n_idx
        """, {"ns1": NNC, "rdfs": RDFS})

    Q3 = prepareQuery("""
        # Get the count of the highest value tags on the primary paths
        SELECT ?l_idx ?n_idx (COUNT(distinct ?tag) as ?number_related_classes) (COUNT(distinct ?class) as ?number_classes_total)
        WHERE {
            ?node ns1:layer ?layer .
            ?layer ns1:layer_index ?l_idx .
            ?node ns1:node_index ?n_idx .
            ?meaning ns1:associated_node ?node .
            ?meaning ns1:name ?mean .
            ?meaning ns1:tag ?tag .
            ?meaning ns1:tag "High" .
            ?end (ns1:secondary_contributor | ns1:primary_contributor)+ ?node .
            ?end ns1:meaning ?class .
            ?class ns1:name ?tag .
        } GROUP BY ?l_idx ?n_idx
        ORDER BY ?l_idx ?n_idx
        """, {"ns1": NNC, "rdfs": RDFS})

    Q4 = prepareQuery("""
        # Get the count of the highest value tags on the primary paths
        SELECT ?l_idx ?n_idx (COUNT(distinct ?tag) as ?number_related_classes)
        WHERE {
            ?node ns1:layer ?layer .
            ?layer ns1:layer_index ?l_idx .
            ?node ns1:node_index ?n_idx .
            ?meaning ns1:associated_node ?node .
            ?meaning ns1:name ?mean .
            ?meaning ns1:tag ?tag .
            ?meaning ns1:tag "High" .
            ?end ns1:meaning ?class .
            ?class ns1:training_percent ?training .
            FILTER(?training > 0.5)
            ?end (ns1:secondary_contributor | ns1:primary_contributor)+ ?node .
            ?class ns1:name ?tag .
        } GROUP BY ?l_idx ?n_idx
        ORDER BY ?l_idx ?n_idx
        """, {"ns1": NNC, "rdfs": RDFS})

    Q5 = prepareQuery("""
        # Number of high values that are on the same node per layer
        SELECT ?l_idx ?n_idx (COUNT(distinct ?mean) as ?number_meanings_for_node)
        WHERE {
            ?node ns1:layer ?layer .
            ?layer ns1:layer_index ?l_idx .
            ?node ns1:node_index ?n_idx .
            ?meaning ns1:associated_node ?node .
            ?meaning ns1:tag "High" .
            ?node ns1:meaning ?mean .
            ?mean ns1:mean_type ns1:by_data .
        } GROUP BY ?l_idx ?n_idx
        ORDER BY ?l_idx ?n_idx
        """, {"ns1": NNC, "rdfs": RDFS})

from typing import Literal
# These should stay the same for each model so I am just going to cache them instead of rebuilding.
global dataset, datasets
dataset = None
datasets = None


def get_model_and_datasets(csv_row: str | int = "0", csv_file: str = "results/BigModel(toOntology).csv"):
    print("Loading model and splitting dataset")
    config = src.cfg.ConfigObject()
    csv_row = str(csv_row)
    config("FromSaveLocation", f"csv {csv_row}" if csv_row.isdigit() else csv_row)
    config.readOnly.remove("ResultsPath")
    config.writeOnce.append("ResultsPath")
    config("ResultsPath", csv_file)
    config("PruningSelection", "Reset")
    load = src.standardLoad(existing_config=config, index=0, structure_only=False)
    global dataset, datasets
    if not dataset or not datasets:
        print("Building datasets from config")
        train, dataset = src.getdata.get_train_test(load["config"])
        dataset = dataset.base
        datasets = src.getdata.split_by_class(src.getdata.get_dataloader(load["config"], train), [x for x in range(dataset.number_of_classes)], individual=True, config=load["config"])
        print("Done building datasets")
    model = src.modelstruct.getModel(load["config"])
    model.cfg = load["config"]
    return model, dataset, datasets


def add_layer(g: "rdflib.Graph", layer_name: str, layer_index: int, number_of_nodes: int | None = None, model: "rdflib.Node" = None):
    layer = rdflib.BNode()
    g.add((layer, RDF.type, NNC.layer))
    g.add((layer, RDFS.label, rdflib.Literal(layer_name)))
    g.add((layer, NNC.layer_index, rdflib.Literal(layer_index)))
    if number_of_nodes:
        g.add((layer, NNC.num_nodes, rdflib.Literal(number_of_nodes)))
    if model:
        g.add((layer, NNC.model, model))
    return layer


def add_node(g: "rdflib.Graph", layer: "rdflib.Node", node_index: int):
    n = rdflib.BNode()
    g.add((n, RDF.type, NNC.node))
    g.add((n, NNC.layer, layer))
    g.add((n, NNC.node_index, rdflib.Literal(node_index)))
    return n


def add_meaning(g: "rdflib.Graph", node: "rdflib.Node", meaning: str, meaning_type: Literal["By Definition", "By Data", "By Inference"] = "By Definition"):
    meaning_types = {
        "By Definition": NNC.by_definition,
        "By Data": NNC.by_data,
        "By Inference": NNC.by_inference
    }
    m = rdflib.BNode()
    g.add((m, RDF.type, NNC.meaning))
    g.add((m, NNC.associated_node, node))
    g.add((node, NNC.meaning, m))
    g.add((m, NNC.name, rdflib.Literal(meaning)))
    g.add((m, NNC.mean_type, meaning_types[meaning_type]))
    return m


def add_node_connection(g: "rdflib.Graph", output_node: "rdflib.Node", weight: "torch.Tensor", input_node: "rdflib.Node"):
    # Note, connections go backwards through the model.
    # output_node should be on the layer after input_node.
    w = weight.item() if isinstance(weight, torch.Tensor) else weight
    c = rdflib.BNode()
    g.add((c, RDF.type, NNC.connection))
    g.add((c, NNC.weight, rdflib.Literal(w)))
    g.add((c, NNC.node, output_node))
    g.add((c, NNC.connected_to, input_node))
    return c


def setup_input_layer(g: "rdflib.Graph", dataset: "src.getdata.BaseDataset", model: "rdflib.Graph" = None):
    input_layer = add_layer(g, "input", -1, number_of_nodes=dataset.number_of_features, model=model)
    dataset_columns = list(dataset.feature_labels.values())
    last_layer = []
    for inputNum in range(dataset.number_of_features):
        # Defining the 'node'
        i = add_node(g, input_layer, inputNum)
        g.add((i, RDF.type, NNC.input_node))

        # Giving names to the inputs
        add_meaning(g, i, dataset_columns[inputNum], "By Definition")

        last_layer.append(i)
    return last_layer


def add_model_layer(g: "rdflib.Graph", layer: "rdflib.Node", module: "torch.nn.modules.Linear", last_layer: list["rdflib.Node"], random_: bool):
    # Define a layer of the network
    this_layer = []
    for num, node in enumerate(module.weight):
        # Define a single node in the layer
        this_layer.append(add_node(g, layer, num))

        max_connection = None
        second_max = None
        for connected_to, connection_weight in enumerate(node):
            if random_:
                connection_weight = src.torch.tensor((2*random()) - 1)

            if abs(connection_weight.item()) > 0.001:
                # Only record connections that are non-zeroed (have a little leeway for pruning methods that use training loop to prune)
                add_node_connection(g, this_layer[-1], connection_weight, last_layer[connected_to])

                # Keep track of best 2 connections
                if max_connection is None or max_connection[0] < abs(connection_weight.item()):
                    if second_max is None or (max_connection and max_connection[0] > second_max[0]):
                        second_max = max_connection
                    max_connection = (abs(connection_weight.item()), connected_to)
                elif second_max is None or second_max[0] < abs(connection_weight.item()):
                    second_max = (abs(connection_weight.item()), connected_to)

        # Add primary and secondary contributors
        if max_connection and len(last_layer) > max_connection[1]:
            g.add((this_layer[-1], NNC.primary_contributor, last_layer[max_connection[1]]))
        if second_max and len(last_layer) > second_max[1]:
            assert last_layer[max_connection[1]] != last_layer[second_max[1]]
            g.add((this_layer[-1], NNC.secondary_contributor, last_layer[second_max[1]]))

    module.graph_nodes = this_layer


def add_model_high_values(g: "rdflib.Graph", datasets: list["src.getdata.BaseDataset"], model: "src.modelstruct.BaseDetectionModel", random_: bool):
    for class_num, dl in enumerate(datasets):
        avg_hook = extramodules.Get_Average_Hook()
        remover = torch.nn.modules.module.register_module_forward_hook(avg_hook)
        # Get the average correct guesses
        s = 0
        for batch in dl:
            model_out = model(batch[0])
            s += sum(torch.argmax(model_out, dim=1) == class_num).item()
        g.add((dl.m, NNC.training_percent, rdflib.Literal(s/len(dl))))

        for mod_name, module in model.named_modules():
            if "fc" in mod_name and not isinstance(module, extramodules.Nothing_Module):
                _, avg = avg_hook.dict[module]
                if random_:
                    avg = torch.rand_like(avg)

                if max(avg) == min(avg):
                    # No max
                    continue

                # Add highest node in the layer
                connection = add_meaning(g, module.graph_nodes[torch.argmax(avg)], "Highest " + datasets[0].base.classes[class_num], "By Data")
                g.add((connection, NNC.tag, rdflib.Literal(datasets[0].base.classes[class_num])))
                g.add((connection, NNC.tag, rdflib.Literal("High")))

                # Add lowest node in the layer
                connection = add_meaning(g, module.graph_nodes[torch.argmax(-avg)], "Lowest " + datasets[0].base.classes[class_num], "By Data")
                g.add((connection, NNC.tag, rdflib.Literal(datasets[0].base.classes[class_num])))
                g.add((connection, NNC.tag, rdflib.Literal("Low")))

                # Remove highest
                avg[torch.argmax(avg)] = torch.min(avg)

                if max(avg) == min(avg):
                    # No second max
                    continue

                # Add second highest node in the layer (by virtue of the highest having been removed)
                connection = add_meaning(g, module.graph_nodes[torch.argmax(avg)], "Second " + datasets[0].base.classes[class_num], "By Data")
                g.add((connection, NNC.tag, rdflib.Literal(datasets[0].base.classes[class_num])))
                g.add((connection, NNC.tag, rdflib.Literal("High")))
        remover.remove()
        del avg_hook


def build_base_facts(csv_row: str | int = "0", csv_file: str = "results/BigModel(toOntology).csv", random_=False):
    model, dataset, datasets = get_model_and_datasets(csv_row, csv_file)

    layer = None
    last_layer = []

    if not random_:
        filename = f"datasets/ontologies/model({csv_file.split('/')[-1]} {csv_row}).ttl"
    else:
        filename = f"datasets/ontologies/random_model ({csv_row}).ttl"

    print("Creating graph")
    g = rdflib.Graph()
    g.bind("", NNC)
    mod = rdflib.BNode()
    g.add((mod, NNC.filename, rdflib.Literal(filename)))
    g.add((mod, RDF.type, NNC.model))
    g.add((mod, NNC.prune_type, rdflib.Literal(model.cfg("PruningSelection")) if not random_ else rdflib.Literal("RandomConnections")))
    g.add((mod, NNC.csv_row, rdflib.Literal(csv_row)))

    count = 0

    # setup for the initial input values to relate the first layer to the input vector
    last_layer = setup_input_layer(g, dataset, mod)

    for mod_name, module in model.named_modules():
        if "fc" in mod_name and not isinstance(module, extramodules.Nothing_Module):
            layer = add_layer(g, mod_name, count, number_of_nodes=len(module.weight), model=mod)
            add_model_layer(g, layer, module, last_layer, random_)
            last_layer = module.graph_nodes
            count += 1

    for attack_type, associated_final_node in enumerate(last_layer):
        m = add_meaning(g, associated_final_node, dataset.classes[attack_type], "By Definition")
        g.add((m, RDF.type, NNC.class_node))
        datasets[attack_type].m = m

    add_model_high_values(g=g, datasets=datasets, model=model, random_=random_)

    g.serialize(filename, encoding="UTF-8")
    print("Created graph")
    return filename, g


def run_really_long_query(file: str = "datasets/model.ttl", graph: "rdflib.Graph" = None):
    if not graph:
        g = rdflib.Graph()
        g.parse(file)
        print("Loaded graph")
    else:
        g = graph
        print("Using existing graph for queries")

    a = next(iter(g.query(QINFO)))
    csv_row_number, pruning_type, total_classes, reduced_classes = a.csv_row_number, a.pruning_type, a.num_classes_total, a.num_classes_reduced

    a = g.query(Q1)
    with open("results/top_down_connections.csv", mode="a") as f:
        print(f'"{file}",,,,', file=f)
        print("Layer,Info,Number of connected,csv row,pruning type", file=f)
        for row in a:
            print(f"{row.l_idx},{row.input},{row.num_paths},{csv_row_number},{pruning_type}", file=f)
        print(',,,,', file=f)
    print("Saved first query results")

    a = g.query(Q2)
    with open("results/bottom_up_connections.csv", mode="a") as f:
        print(f'"{file}",,,,', file=f)
        print("Layer,Info,Number of connected,csv row,pruning type", file=f)
        for row in a:
            print(f"{row.l_idx},{row.input},{row.num_paths},{csv_row_number},{pruning_type}", file=f)
        print(',,,,', file=f)
    print("Saved second query results")

    a = g.query(Q3)
    with open("results/high_nodes_along_connections.csv", mode="a") as f:
        print(f'"{file}",,,,,', file=f)
        print("Layer,node,Number of connected classes,Number of classes total,csv row,pruning type", file=f)
        for row in a:
            print(f"{row.l_idx},{row.n_idx},{row.number_related_classes},{total_classes},{csv_row_number},{pruning_type}", file=f)
        print(',,,,,', file=f)
    print("Saved third query results")

    a = g.query(Q4)
    with open("results/high_nodes_of_reduced_classes.csv", mode="a") as f:
        print(f'"{file}",,,,,', file=f)
        print("Layer,node,Number of connected classes,Number of classes total,csv row,pruning type", file=f)
        for row in a:
            print(f"{row.l_idx},{row.n_idx},{row.number_related_classes},{reduced_classes},{csv_row_number},{pruning_type}", file=f)
        print(',,,,,', file=f)
    print("Saved fourth query results")

    a = g.query(Q5)
    with open("results/high_nodes.csv", mode="a") as f:
        print(f'"{file}",,,,', file=f)
        print("Layer,node,Number of meanings for node,csv row,pruning type", file=f)
        for row in a:
            print(f"{row.l_idx},{row.n_idx},{row.number_meanings_for_node},{csv_row_number},{pruning_type}", file=f)
        print(',,,,', file=f)
    print("Saved fifth query results")


def make_pivot_table_from_top_down_connections():
    df: pd.DataFrame = pd.read_csv("results/top_down_connections.csv", header=False, columns=["Layer", "Extra_info", "Num_connections"])

    table = df.pivot_table(values="Num_connections", index="Layer", columns="Num_connections", aggfunc="count")

    table.to_latex("results/layer_connections.txt")
    return table


def select_best_rows(csv_file: str = "results/BigModel(toOntology).csv"):
    # Wanted to make this automatic but did not eventually do that.
    df = pd.read_csv("results/BigModel(toOntology).csv")
    # df = df[df["WeightPrunePercent"] == "[0.62, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 1.0]"]
    df = df[df["Notes"] == 0]
    return list(df.index)


if __name__ == "__main__":
    if not False:
        for x in select_best_rows():
            print(f"running for csv row {x}")
            if not os.path.exists(f"datasets/ontologies/model(BigModel(toOntology).csv {x}).ttl"):
                path, g = build_base_facts(random_=False, csv_row=f"{x}")
                run_really_long_query(f"datasets/ontologies/model(BigModel(toOntology).csv {x}).ttl", graph=g)
            else:
                pass
                run_really_long_query(f"datasets/ontologies/model(BigModel(toOntology).csv {x}).ttl")
        for x in [0, 1, 2]:
            print(f"running for random {x}")
            if not os.path.exists(f"datasets/ontologies/random_model ({x}).ttl"):
                path, g = build_base_facts(random_=True, csv_row=f"{x}")
                run_really_long_query(f"datasets/ontologies/random_model ({x}).ttl", graph=g)
            else:
                pass
                run_really_long_query(f"datasets/ontologies/random_model ({x}).ttl")
    else:
        print(select_best_rows())
