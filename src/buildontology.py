from typing import Literal, TYPE_CHECKING
import os
if __name__ == "__main__" or TYPE_CHECKING:
    import sys
    from random import random
    # from functools import cache

    import pandas as pd
    import rdflib
    import torch.nn
    from rdflib.namespace import RDF, RDFS
    from rdflib.plugins.sparql import prepareQuery

    import __init__ as src
    import extramodules

    RESULTS_PATH = "results"
    RESULTS_FILE = "BigModel(toOntology).csv"

    if len(sys.argv) > 1 and len(sys.argv[1]) > 0:
        RESULTS_FILE = sys.argv[1]
        RESULTS_FILE = RESULTS_FILE.removeprefix(RESULTS_PATH).removeprefix("/")
    RESULTS_FILE = "temp.csv"

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
    q1_path = os.path.join(RESULTS_PATH, "top_down_connections.csv")

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
    q2_path = os.path.join(RESULTS_PATH, "bottom_up_connections.csv")

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
    q3_path = os.path.join(RESULTS_PATH, "high_nodes_along_connections.csv")

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
    q4_path = os.path.join(RESULTS_PATH, "high_nodes_of_reduced_classes.csv")

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
    q5_path = os.path.join(RESULTS_PATH, "high_nodes.csv")

    # Q6 = prepareUpdate("""
    #     INSERT {
    #         ?from ns1:LRP ?relevance
    #     }
    #     WHERE {
    #         ?node ns1:layer ?layer .
    #         ?layer ns1:layer_index ?l_idx .
    #         ?node ns1:node_index ?n_idx .
    #         ?node ns1:meaning ?class .
    #         ?connection ns1:node ?node .
    #         ?connection ns1:connected_to ?from .
    #         ?connection ns1:weight ?val .
    #         ?class ns1:training_percent ?training .
    #         SELECT (SUM(?val) AS ?layer_relevance)
    #         WHERE {
    #             ?node ns1:layer ?layer .
    #             ?layer ns1:layer_index ?l_idx .
    #             ?node ns1:node_index ?n_idx .
    #             ?node ns1:meaning ?class .
    #             ?connection ns1:node ?node .
    #             ?connection ns1:connected_to ?from .
    #             ?connection ns1:weight ?val .
    #             ?class ns1:training_percent ?training .
    #         } GROUPBY ?l_idx
    #         BIND((?val / ?layer_relevance) AS ?relevance)
    #     }
    #     """, {"ns1": NNC, "rdfs": RDFS})
else:
    RESULTS_PATH = ""
    RESULTS_FILE = ""


# These should stay the same for each model so I am just going to cache them instead of rebuilding.
global dataset, datasets
dataset = None
datasets = None


if TYPE_CHECKING:
    NNC.filename = "test"


# @cache
def get_model_and_datasets(csv_row: str | int = "0", csv_file: str = os.path.join(RESULTS_PATH, RESULTS_FILE)) -> tuple["src.modelstruct.BaseDetectionModel", "src.getdata.ModifiedDataloader", list["src.getdata.ModifiedDataloader"]]:
    """
    Loads the model and the dataset from a specific csv file and row.

    Args:
        csv_row (str | int, optional): Row of the csv file to retrieve the model from. Defaults to "0".
        csv_file (str, optional): Csv file to read. Defaults to "results/BigModel(toOntology).csv".

    Returns:
        src.modelstruct.BaseDetectionModel: Model that has been loaded.
        src.getdata.ModifiedDataloader: Dataloader to read for general data.
        list:
            src.getdata.ModifiedDataloader: Dataloaders sorted by class.
    """
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


def get_waypoint_model(model: "src.modelstruct.BaseDetectionModel", waypoint_number: int, csv_row: str | int = "0", csv_file: str = os.path.join(RESULTS_PATH, "BigModel(OntologyWaypoints).csv")):
    if csv_row.isnumeric():
        csv_row = int(csv_row)
    row = pd.read_csv(csv_file).iloc[csv_row]
    waypoint_pth = row[f"-Waypoint_{waypoint_number}-"]
    waypoint_loss = row[f"Epoch waypoint {waypoint_number} mean_loss"]
    model.load_state_dict(torch.load(os.path.join("savedModels", "waypoints", waypoint_pth), map_location=model.cfg("Device")))
    model.cfg("PruningSelection", f"Waypoint_{waypoint_number}")

    return model, waypoint_loss


def modify_model_pruning_selection(model: "src.modelstruct.BaseDetectionModel"):
    """
    Adds information about the weight prune percent to the pruning selection so that we don't need to modify the query structure to pass it through.

    Args:
        model (src.modelstruct.BaseDetectionModel): Model that we want to modify the pruning selection field for.
    """
    # Prune selection field is always appended.
    if isinstance(model.cfg("WeightPrunePercent"), str):
        model.cfg("PruningSelection", model.cfg("WeightPrunePercent").split(",")[0])
    else:
        model.cfg("PruningSelection", str(model.cfg("WeightPrunePercent")[0]))


def add_layer(g: "rdflib.Graph", layer_name: str, layer_index: int, number_of_nodes: int | None = None, model: "rdflib.Node" = None) -> "rdflib.Node":
    """
    Adds a layer to an rdflib graph. Where a layer is an entity that contains several nodes.

    Args:
        g (rdflib.Graph): Graph to add the layer to.
        layer_name (str): Name the layer.
        layer_index (int): Numerical index to describe the layer.
        number_of_nodes (int | None, optional): Specify the number of nodes in the layer. Defaults to None.
        model (rdflib.Node, optional): Model entity in the graph. Defaults to None.

    Returns:
        rdflib.Node: The layer entity node that was added.
    """
    layer = rdflib.BNode()
    g.add((layer, RDF.type, NNC.layer))
    g.add((layer, RDFS.label, rdflib.Literal(layer_name)))
    g.add((layer, NNC.layer_index, rdflib.Literal(layer_index)))
    if number_of_nodes:
        g.add((layer, NNC.num_nodes, rdflib.Literal(number_of_nodes)))
    if model:
        g.add((layer, NNC.model, model))
    return layer


def add_node(g: "rdflib.Graph", layer: "rdflib.Node", node_index: int) -> "rdflib.Node":
    """
    Add an individual neuron node to a specified layer in a rdflib graph

    Args:
        g (rdflib.Graph): Graph to add the node to.
        layer (rdflib.Node): Layer to add the node to.
        node_index (int): Index of the node within the layer.

    Returns:
        rdflib.Node: The neural node entity node that was added.
    """
    n = rdflib.BNode()
    g.add((n, RDF.type, NNC.node))
    g.add((n, NNC.layer, layer))
    g.add((n, NNC.node_index, rdflib.Literal(node_index)))
    return n


def add_meaning(g: "rdflib.Graph", node: "rdflib.Node", meaning: str, meaning_type: Literal["By Definition", "By Data", "By Inference"] = "By Definition") -> "rdflib.Node":
    """
    Adds additional meaning to a given node of the nn. Either by definition for input and output neurons, by data for observed from the dataset, and by inference for other.

    Args:
        g (rdflib.Graph): Graph to add the meaning to.
        node (rdflib.Node): Specific node to give more info for.
        meaning (str): The information to add to the node.
        meaning_type (Literal[&quot;By Definition&quot;, &quot;By Data&quot;, &quot;By Inference&quot;], optional): Type of meaning. By Definition should be values that are defined purely on how the model is structured. By Data is observed from the dataset. And Inference is created from elsewhere. Defaults to "By Definition".

    Returns:
        rdflib.Node: The rdflib node that has been added to the graph.
    """
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


def add_node_connection(g: "rdflib.Graph", output_node: "rdflib.Node", weight: "torch.Tensor", input_node: "rdflib.Node") -> "rdflib.Node":
    """
    Add a connection entity between a node and another node of the previous layer.

    Args:
        g (rdflib.Graph): Graph to add the connection to.
        output_node (rdflib.Node): Node that is further along in the neural network
        weight (torch.Tensor): Connection multiplier
        input_node (rdflib.Node): Node that is less far along on the neural network.

    Returns:
        rdflib.Node: Connection entity that has been added to the graph.
    """
    # Note, connections go backwards through the model.
    # output_node should be on the layer after input_node.
    w = weight.item() if isinstance(weight, torch.Tensor) else weight
    c = rdflib.BNode()
    g.add((c, RDF.type, NNC.connection))
    g.add((c, NNC.weight, rdflib.Literal(w)))
    g.add((c, NNC.node, output_node))
    g.add((c, NNC.connected_to, input_node))
    return c


def setup_input_layer(g: "rdflib.Graph", dataset: "src.getdata.BaseDataset", model: "rdflib.Graph" = None) -> list["rdflib.Node"]:
    """
    Adds a layer that serves as the input layer that is specially marked with all nodes having By Definition meanings.

    Args:
        g (rdflib.Graph): graph to modify
        dataset (src.getdata.BaseDataset): dataset to grab the meaning names from.
        model (rdflib.Graph, optional): model entity in the graph. Defaults to None.

    Returns:
        list:
            rdflib.Node: Node entities in the input layer.
    """
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
    """
    Add all of the connections to a layer from a prior layer. (Different from add_layer which just adds a single layer entity.)

    Args:
        g (rdflib.Graph): Graph to add a model to.
        layer (rdflib.Node): Layer entity to modify by adding all connections.
        module (torch.nn.modules.Linear): Module associated with this layer.
        last_layer (list[&quot;rdflib.Node&quot;]): list of rdflib node entities that are the nodes in the last layer.
        random_ (bool): whether the connections should be assigned randomly for reference.
    """
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
    """
    Add the high values from the dataset to the graph as meaning to the nodes.

    Args:
        g (rdflib.Graph): graph to be modifying.
        datasets (list[&quot;src.getdata.BaseDataset&quot;]): datasets to test, split up by class.
        model (src.modelstruct.BaseDetectionModel): model to test.
        random_ (bool): whether the connections should be assigned randomly for reference.
    """
    for class_num, dl in enumerate(datasets):
        avg_hook = extramodules.Get_Average_Hook()
        remover = torch.nn.modules.module.register_module_forward_hook(avg_hook)
        # Get the average correct guesses
        s = 0
        # print(len(dl))
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


def build_base_facts(csv_row: str | int = "0", csv_file: str = os.path.join(RESULTS_PATH, RESULTS_FILE), random_=False, waypoint=None, save=True, add_prune_info=False) -> tuple[str, "rdflib.Graph"]:
    """
    Generate the model and info without any inference.

    Args:
        csv_row (str | int, optional): Row of the csv to read or specific file if it is a path, works the same as csv row command line arg. Defaults to "0".
        csv_file (str, optional): File to read for the csv row. Defaults to "results/BigModel(toOntology).csv".
        random_ (bool, optional): whether to set values randomly for comparison. Defaults to False.

    Returns:
        str: Name of file that the graph was saved in.
        rdflib.graph: Final built graph for immediate use.
    """
    model, dataset, datasets = get_model_and_datasets(csv_row, csv_file)
    if waypoint is not None:
        model, _ = get_waypoint_model(model, waypoint, csv_row, csv_file)
    if add_prune_info:
        modify_model_pruning_selection(model)

    layer = None
    last_layer = []

    if not random_:
        filename = os.path.join("datasets", "ontologies", f"model({csv_file.split('/')[-1]} {csv_row}).ttl")
    else:
        filename = os.path.join("datasets", "ontologies", f"random_model ({csv_row}).ttl")

    print("Creating graph")
    g = rdflib.Graph()
    g.bind("", NNC)
    mod = rdflib.BNode()
    g.add((mod, NNC.filename, rdflib.Literal(filename)))
    g.add((mod, RDF.type, NNC.model))
    g.add((mod, NNC.prune_type, rdflib.Literal(model.cfg("PruningSelection")) if not random_ else rdflib.Literal("RandomConnections")))
    g.add((mod, NNC.csv_row, rdflib.Literal(csv_row)))
    print("Test1")

    count = 0

    # setup for the initial input values to relate the first layer to the input vector
    last_layer = setup_input_layer(g, dataset, mod)

    print("Test2")
    for mod_name, module in model.named_modules():
        if "fc" in mod_name and not isinstance(module, extramodules.Nothing_Module):
            layer = add_layer(g, mod_name, count, number_of_nodes=len(module.weight), model=mod)
            add_model_layer(g, layer, module, last_layer, random_)
            last_layer = module.graph_nodes
            count += 1
    print("Test3")

    for attack_type, associated_final_node in enumerate(last_layer):
        m = add_meaning(g, associated_final_node, dataset.classes[attack_type], "By Definition")
        g.add((m, RDF.type, NNC.class_node))
        datasets[attack_type].m = m

    print("Test4")
    add_model_high_values(g=g, datasets=datasets, model=model, random_=random_)

    print("Test5")
    if save:
        g.serialize(filename, encoding="UTF-8")
    print("Created graph")
    return filename, g


def run_really_long_query(file: str = os.path.join("datasets", "model.ttl"), graph: "rdflib.Graph" = None):
    """
    Runs all of the really long queries. Each query is added to the end of its own file.

    Args:
        file (str, optional): File to load from if graph is none. Defaults to "datasets/model.ttl".
        graph (rdflib.Graph, optional): Graph to run the queries on. Defaults to None.
    """
    if not graph:
        g = rdflib.Graph()
        g.parse(file)
        print("Loaded graph")
    else:
        g = graph
        print("Using existing graph for queries")

    a = next(iter(g.query(QINFO)))
    csv_row_number, pruning_type, total_classes, reduced_classes = a.csv_row_number, a.pruning_type, a.num_classes_total, a.num_classes_reduced

    if Q1 is not None:
        a = g.query(Q1)
        with open(q1_path, mode="a") as f:
            print(f'"{file}",,,,', file=f)
            print("Layer,Info,Number of connected,csv row,pruning type", file=f)
            for row in a:
                print(f"{row.l_idx},{row.input},{row.num_paths},{csv_row_number},{pruning_type}", file=f)
            print(',,,,', file=f)
        print("Saved first query results")

    if Q2 is not None:
        a = g.query(Q2)
        with open(q2_path, mode="a") as f:
            print(f'"{file}",,,,', file=f)
            print("Layer,Info,Number of connected,csv row,pruning type", file=f)
            for row in a:
                print(f"{row.l_idx},{row.input},{row.num_paths},{csv_row_number},{pruning_type}", file=f)
            print(',,,,', file=f)
        print("Saved second query results")

    if Q3 is not None:
        a = g.query(Q3)
        with open(q3_path, mode="a") as f:
            print(f'"{file}",,,,,', file=f)
            print("Layer,node,Number of connected classes,Number of classes total,csv row,pruning type", file=f)
            for row in a:
                print(f"{row.l_idx},{row.n_idx},{row.number_related_classes},{total_classes},{csv_row_number},{pruning_type}", file=f)
            print(',,,,,', file=f)
        print("Saved third query results")

    if Q4 is not None:
        a = g.query(Q4)
        with open(q4_path, mode="a") as f:
            print(f'"{file}",,,,,', file=f)
            print("Layer,node,Number of connected classes,Number of classes total,csv row,pruning type", file=f)
            for row in a:
                print(f"{row.l_idx},{row.n_idx},{row.number_related_classes},{reduced_classes},{csv_row_number},{pruning_type}", file=f)
            print(',,,,,', file=f)
        print("Saved fourth query results")

    if Q5 is not None:
        a = g.query(Q5)
        with open(q5_path, mode="a") as f:
            print(f'"{file}",,,,', file=f)
            print("Layer,node,Number of meanings for node,csv row,pruning type", file=f)
            for row in a:
                print(f"{row.l_idx},{row.n_idx},{row.number_meanings_for_node},{csv_row_number},{pruning_type}", file=f)
            print(',,,,', file=f)
        print("Saved fifth query results")

    # g.update(Q6)
    del g


def make_pivot_table_from_top_down_connections() -> "pd.DataFrame":
    """
    Generates a pivot table to observe the top down connections from the most recent file generated.

    Returns:
        pd.DataFrame: the pivot table generated.
    """
    df: pd.DataFrame = pd.read_csv(os.path.join(RESULTS_PATH, "top_down_connections.csv"), header=False, columns=["Layer", "Extra_info", "Num_connections"])

    table = df.pivot_table(values="Num_connections", index="Layer", columns="Num_connections", aggfunc="count")

    table.to_latex(os.path.join(RESULTS_PATH, "layer_connections.txt"))
    return table


def select_best_rows(csv_file: str = os.path.join(RESULTS_PATH, RESULTS_FILE), original_row: None | int = None) -> list[int]:
    # Wanted to make this automatic but did not eventually do that.
    df = pd.read_csv(csv_file)
    # df = df[df["WeightPrunePercent"] == "[0.62, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 1.0]"]
    if original_row is not None:
        df = df[df["AssociatedOriginalRow"].notna()]
        df = df[df["AssociatedOriginalRow"].astype(int).astype(str) == original_row]

    df = df[df["Notes"] == 0]
    return list(df.index)


def run_standard_tests():
    for x in [q1_path, q2_path, q3_path, q4_path, q5_path]:
        if os.path.exists(x):
            os.remove(x)
    for x in select_best_rows():
        print(f"running for csv row {x}")
        # Checking for pre-created ontology
        if not os.path.exists(os.path.join("datasets", "ontologies", f"model({RESULTS_FILE} {x}).ttl")):
            path, g = build_base_facts(random_=False, csv_row=f"{x}")
            run_really_long_query(os.path.join("datasets", "ontologies", f"model({RESULTS_FILE} {x}).ttl"), graph=g)
            del g
        else:
            print("Loading the file: " + os.path.join("datasets", "ontologies", f"model({RESULTS_FILE} {x}).ttl"))
            run_really_long_query(os.path.join("datasets", "ontologies", f"model({RESULTS_FILE} {x}).ttl"))


def run_standard_random_baseline_tests():
    for x in [0, 1, 2]:  # first three rows should be the initial models
        print(f"running for random {x}")
        if not os.path.exists(os.path.join("datasets", "ontologies", f"random_model ({x}).ttl")):
            path, g = build_base_facts(random_=True, csv_row=f"{x}")
            run_really_long_query(os.path.join("datasets", "ontologies", f"random_model ({x}).ttl"), graph=g)
        else:
            pass
            run_really_long_query(os.path.join("datasets", "ontologies", f"random_model ({x}).ttl"))


def run_waypoint_tests():
    global q1_path, Q2, q3_path, Q4, Q5
    q1_path = os.path.join(RESULTS_PATH, "waypointing", "top_down_connections.csv")
    Q2 = None
    q3_path = os.path.join(RESULTS_PATH, "waypointing", "high_nodes_along_connections.csv")
    Q4 = None
    Q5 = None
    print(repr(sys.argv[1]))
    for x in range(13):
        print(f"Starting {x} on row {repr(sys.argv[1])}")
        path, g = build_base_facts(csv_file=os.path.join(RESULTS_PATH, "BigModel(OntologyWaypoints).csv"), random_=False, csv_row=f"{sys.argv[1]}", waypoint=x, save=False)
        run_really_long_query("", graph=g)


def run_pruning_tests():
    global q1_path, Q2, q3_path, Q4, Q5
    q1_path = os.path.join(RESULTS_PATH, "pruning", "top_down_connections.csv")
    Q2 = None
    q3_path = os.path.join(RESULTS_PATH, "pruning", "high_nodes_along_connections.csv")
    Q4 = None
    Q5 = None
    print(repr(sys.argv[1]))
    for x in select_best_rows(os.path.join(RESULTS_PATH, "BigModel(OntologyPruning).csv"), original_row=sys.argv[1]):
        print(f"Starting {x} on original row {repr(sys.argv[1])}")
        path, g = build_base_facts(csv_file=os.path.join(RESULTS_PATH, "BigModel(OntologyPruning).csv"), random_=False, csv_row=f"{x}", save=False, add_prune_info=True)
        run_really_long_query("", graph=g)
        del g


if __name__ == "__main__":
    if True:
        run_standard_tests()
        run_standard_random_baseline_tests()
    elif False:
        print("Building ontologies and running the queries based on the waypoint columns on the line of argv $1")
        run_waypoint_tests()
    else:
        print("Building ontologies  and running the queries on the models that were created by pruning the row argv $1. So it runs on the pruned models.")
        run_pruning_tests()
