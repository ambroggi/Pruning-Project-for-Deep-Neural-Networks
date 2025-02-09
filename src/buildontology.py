import os
from random import random

import pandas as pd
import rdflib
import torch.nn
from rdflib.namespace import RDF, RDFS

import __init__ as src
import extramodules

RDFS
NNC = rdflib.Namespace("https://github.com/ambroggi/Pruning-Project-for-Deep-Neural-Networks")   # Neural Network Connections (I guess I should have a location that is not example.org?)


def build_base_facts(csv_row: str | int = "0", csv_file: str = "results/BigModel(v0.131).csv", random_=False):
    config = src.cfg.ConfigObject.get_param_from_args()
    csv_row = str(csv_row)
    config("FromSaveLocation", f"csv {csv_row}" if csv_row.isdigit() else csv_row)
    config.readOnly.remove("ResultsPath")
    config.writeOnce.append("ResultsPath")
    config("ResultsPath", csv_file)
    config("PruningSelection", "Reset")
    load = src.standardLoad(existing_config=config, index=0, structure_only=False)
    train, dataset = src.getdata.get_train_test(load["config"])
    dataset = dataset.base
    datasets = src.getdata.split_by_class(src.getdata.get_dataloader(load["config"], train), [x for x in range(dataset.number_of_classes)], individual=True, config=load["config"])
    model = src.modelstruct.getModel(load["config"])
    layer = None
    last_layer = []
    this_layer = []

    if not random_:
        filename = f"datasets/model({csv_file.split('/')[-1]} {csv_row}).ttl"
    else:
        filename = f"datasets/random_model ({csv_row}).ttl"

    g = rdflib.Graph()
    g.bind("", NNC)
    g.add((rdflib.Literal(filename), NNC.filename, rdflib.Literal(filename)))

    count = 0

    # Define the input vector layer of the network
    input_layer = rdflib.BNode()
    g.add((input_layer, RDF.type, NNC.layer))
    g.add((input_layer, RDFS.label, rdflib.Literal("Input")))
    g.add((input_layer, NNC.layer_index, rdflib.Literal(-1)))

    for mod_name, module in model.named_modules():
        if "fc" in mod_name:
            # Define a layer of the network
            layer = rdflib.BNode()
            g.add((layer, RDF.type, NNC.layer))
            g.add((layer, RDFS.label, rdflib.Literal(mod_name)))
            g.add((layer, NNC.layer_index, rdflib.Literal(count)))
            g.add((layer, NNC.num_nodes, rdflib.Literal(len(module.weight))))
            for num, node in enumerate(module.weight):
                # Define a single node in the layer
                n = rdflib.BNode()
                this_layer.append(n)
                g.add((n, RDF.type, NNC.node))
                g.add((n, NNC.layer, layer))
                g.add((n, NNC.node_index, rdflib.Literal(num)))

                # setup for the initial input values to relate the first layer to the input vector
                if len(last_layer) == 0:
                    dataset_columns = list(dataset.feature_labels.values())
                    for inputNum, _ in enumerate(node):
                        # Defining the 'node'
                        i = rdflib.BNode()
                        g.add((i, RDF.type, NNC.node))
                        g.add((i, RDF.type, NNC.input_node))
                        g.add((i, NNC.layer, input_layer))
                        g.add((i, NNC.node_index, rdflib.Literal(inputNum)))

                        # Giving names to the inputs
                        m = rdflib.BNode()
                        g.add((m, RDF.type, NNC.meaning))
                        g.add((m, NNC.associated_node, i))
                        g.add((i, NNC.meaning, m))
                        g.add((m, NNC.name, rdflib.Literal(dataset_columns[inputNum])))

                        last_layer.append(i)
                    g.add((input_layer, NNC.num_nodes, rdflib.Literal(len(node))))

                max_connection = None
                second_max = None
                for connected_to, connec in enumerate(node):
                    if random_:
                        connec = src.torch.tensor((2*random()) - 1)
                    if abs(connec.item()) > 0.001:
                        # if connec.item() > 0.001:
                        c = rdflib.BNode()
                        g.add((c, RDF.type, NNC.connection))
                        g.add((c, NNC.weight, rdflib.Literal(connec.item())))
                        g.add((c, NNC.node, n))

                        if len(last_layer) > connected_to:
                            g.add((c, NNC.connected_to, last_layer[connected_to]))

                        if max_connection is None or max_connection[0] < abs(connec.item()):
                            if second_max is None or (max_connection and max_connection[0] > second_max[0]):
                                second_max = max_connection
                            max_connection = (abs(connec.item()), connected_to)
                        elif second_max is None or second_max[0] < abs(connec.item()):
                            second_max = (abs(connec.item()), connected_to)

                        # if max_connection is None or max_connection[0] < connec.item():
                        #     max_connection = (connec.item(), connected_to)

                if max_connection and len(last_layer) > max_connection[1]:
                    g.add((n, NNC.primary_contributor, last_layer[max_connection[1]]))
                if second_max and len(last_layer) > second_max[1]:
                    assert last_layer[max_connection[1]] != last_layer[second_max[1]]
                    g.add((n, NNC.secondary_contributor, last_layer[second_max[1]]))

            module.graph_nodes = this_layer
            last_layer = this_layer
            this_layer = []
            count += 1

    for attack_type, associated_final_node in enumerate(last_layer):
        atk = rdflib.BNode()
        g.add((atk, RDF.type, NNC.meaning))
        g.add((atk, NNC.associated_node, associated_final_node))
        g.add((associated_final_node, NNC.meaning, atk))
        g.add((atk, NNC.name, rdflib.Literal(dataset.classes[attack_type])))

    for class_num, dl in enumerate(datasets):
        avghook = extramodules.Get_Average_Hook()
        remover = torch.nn.modules.module.register_module_forward_hook(avghook)
        for batch in dl:
            model(batch[0])
        for mod_name, module in model.named_modules():
            if "fc" in mod_name:
                _, avg = avghook.dict[module]
                if random_:
                    avg = torch.rand_like(avg)
                connection = rdflib.BNode()
                g.add((connection, RDF.type, NNC.meaning))
                g.add((connection, NNC.associated_node, module.graph_nodes[torch.argmax(avg)]))
                g.add((connection, NNC.name, rdflib.Literal("Highest " + dataset.classes[class_num])))
                g.add((connection, NNC.tag, rdflib.Literal(dataset.classes[class_num])))
                avg[torch.argmax(avg)] = torch.min(avg)
                connection = rdflib.BNode()
                g.add((connection, RDF.type, NNC.meaning))
                g.add((connection, NNC.associated_node, module.graph_nodes[torch.argmax(avg)]))
                g.add((connection, NNC.name, rdflib.Literal("Second  " + dataset.classes[class_num])))
                g.add((connection, NNC.tag, rdflib.Literal(dataset.classes[class_num])))
        remover.remove()
        del avghook

    g.serialize(filename, encoding="UTF-8")
    return filename, g


def run_really_long_query(file: str = "datasets/model.ttl"):
    g = rdflib.Graph()
    g.parse(file)
    print("Loaded graph")
    q = """
        PREFIX ns1: <https://github.com/ambroggi/Pruning-Project-for-Deep-Neural-Networks>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?l_idx (SAMPLE(?input_meaning) as ?input) (COUNT(DISTINCT ?mean) as ?num_paths)
        WHERE {
            ?node ns1:layer ?layer .
            ?layer ns1:layer_index ?l_idx .
            ?node ns1:node_index ?nidx .
            ?meaning ns1:associated_node/(ns1:secondary_contributor | ns1:primary_contributor)+ ?node .
            ?meaning ns1:name ?mean .
            OPTIONAL {
                ?node ns1:meaning ?in_meaning .
                ?in_meaning ns1:name ?input_meaning .
            }
        } GROUP BY ?nidx ?l_idx
        ORDER BY desc(?l_idx) ?nidx
        """

    a = g.query(q)
    with open("top_down_connections.csv", mode="w") as f:
        print(f'"{file}", , ', file=f)
        print("Layer, Info, Number of connected", file=f)
        for row in a:
            print(f"{row.l_idx}, {row.input}, {row.num_paths}", file=f)
    print("Saved first query results")

    q = """
        PREFIX ns1: <https://github.com/ambroggi/Pruning-Project-for-Deep-Neural-Networks>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        # Get "most important" input numbers counts
        SELECT ?l_idx (SAMPLE(?input_meaning) as ?input) (COUNT(DISTINCT ?mean) as ?num_paths)
        WHERE {
            ?node ns1:layer ?layer .
            ?layer ns1:layer_index ?l_idx .
            ?node ns1:node_index ?nidx .
            ?meaning ns1:associated_node ?start .
            ?node (ns1:secondary_contributor | ns1:primary_contributor)+ ?start .
            ?meaning ns1:name ?mean .
            OPTIONAL {
                ?node ns1:meaning ?in_meaning .
                ?in_meaning ns1:name ?input_meaning .
            }
        } GROUP BY ?nidx ?l_idx
        ORDER BY desc(?l_idx) ?nidx
        """

    a = g.query(q)
    with open("bottom_up_connections.csv", mode="w") as f:
        print(f'"{file}", , ', file=f)
        print("Layer, Info, Number of connected", file=f)
        for row in a:
            print(f"{row.l_idx}, {row.input}, {row.num_paths}", file=f)
    print("Saved second query results")

    q = """
        PREFIX ns1: <https://github.com/ambroggi/Pruning-Project-for-Deep-Neural-Networks>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        # Get the count of the highest value tags on the primary paths
        SELECT ?l_idx ?n_idx (COUNT(distinct ?tag) as ?number_related_classes)
        WHERE {
            ?node ns1:layer ?layer .
            ?layer ns1:layer_index ?l_idx .
            ?node ns1:node_index ?n_idx .
            ?meaning ns1:associated_node ?node .
            ?meaning ns1:name ?mean .
            ?meaning ns1:tag ?tag .
            ?end (ns1:secondary_contributor | ns1:primary_contributor)+ ?node .
            ?end ns1:meaning ?class .
            ?class ns1:name ?tag .
        } GROUP BY ?l_idx ?n_idx
        ORDER BY ?l_idx ?n_idx
        """

    a = g.query(q)
    with open("high_nodes_along_connections.csv", mode="w") as f:
        print(f'"{file}", , ', file=f)
        print("Layer, node, Number of connected classes", file=f)
        for row in a:
            print(f"{row.l_idx}, {row.n_idx}, {row.number_related_classes}", file=f)
    print("Saved third query results")


def make_pivot_table_from_top_down_connections():
    df: pd.DataFrame = pd.read_csv("top_down_connections.csv", header=False, columns=["Layer", "Extra_info", "Num_connections"])

    table = df.pivot_table(values="Num_connections", index="Layer", columns="Num_connections", aggfunc="count")

    table.to_latex("results/layer_connections.txt")
    return table


if __name__ == "__main__":
    for x in [0, 1, 2]:
        if not os.path.exists(f"datasets/model(BigModel(v0.131).csv {x}).ttl"):
            path, g = build_base_facts(random_=False, csv_row=f"{x}")
        run_really_long_query(f"datasets/model(BigModel(v0.131).csv {x}).ttl")
