# from random import randint
import rdflib
from rdflib.namespace import RDF, RDFS
import src

RDFS
NNC = rdflib.Namespace("https://github.com/ambroggi/Pruning-Project-for-Deep-Neural-Networks")   # Neural Network Connections (I guess I should have a location that is not example.org?)


def build_base_facts():
    config = src.cfg.ConfigObject.get_param_from_args()
    config("FromSaveLocation", "csv 0")
    config("PruningSelection", "Reset")
    load = src.standardLoad(existing_config=config, index=0)
    dataset = src.getdata.get_dataset(load["config"]).base
    model = src.modelstruct.getModel(load["config"])
    layer = None
    last_layer = []
    this_layer = []

    g = rdflib.Graph()
    g.bind("", NNC)

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
                    # g.add((n, NNC.primary_contributor, last_layer[randint(0, len(last_layer)-1)]))
                if second_max and len(last_layer) > second_max[1]:
                    assert last_layer[max_connection[1]] != last_layer[second_max[1]]
                    g.add((n, NNC.secondary_contributor, last_layer[second_max[1]]))
                    # g.add((n, NNC.secondary_contributor, last_layer[randint(0, len(last_layer)-1)]))

            last_layer = this_layer
            this_layer = []
            count += 1

    for attack_type, associated_final_node in enumerate(last_layer):
        atk = rdflib.BNode()
        g.add((atk, RDF.type, NNC.meaning))
        g.add((atk, NNC.associated_node, associated_final_node))
        g.add((associated_final_node, NNC.meaning, atk))
        g.add((atk, NNC.name, rdflib.Literal(dataset.classes[attack_type])))

    g.serialize("datasets/model.ttl", encoding="UTF-8")
    # g.serialize("datasets/random_model.ttl", encoding="UTF-8")


def build_pruned_graph():
    g = rdflib.Graph()
    g.parse("datasets/model.ttl")

    g.bind("", NNC)


def run_really_long_query():
    g = rdflib.Graph()
    g.parse("datasets/model.ttl")
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
    with open("out.csv", mode="w") as f:
        for row in a:
            print(f"{row.l_idx}, {row.input}, {row.num_paths}", file=f)

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
    with open("out2.csv", mode="w") as f:
        for row in a:
            print(f"{row.l_idx}, {row.input}, {row.num_paths}", file=f)


if __name__ == "__main__":
    build_base_facts()
    run_really_long_query()

# PREFIX ns1: <https://github.com/ambroggi/Pruning-Project-for-Deep-Neural-Networks>
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

# SELECT (COUNT(?connection) as ?num_conns) (SAMPLE(?pr_n_idx) as ?primary_n) (SAMPLE(?pr_l_idx) as ?primary_l) ?n_idx ?l_idx (COUNT(distinct ?pr_n_idx) as ?count_primary_n) (COUNT(distinct ?pr_l_idx) as ?count_primary_l) WHERE {
#     ?connection ns1:node ?node .
#     ?node ns1:node_index ?n_idx .
#     FILTER (?n_idx < 3)
#     ?node ns1:layer ?layer .
#     ?layer ns1:layer_index ?l_idx .
#     OPTIONAL {
#         ?node ns1:primary_contributor ?pr_con .
#         ?pr_con ns1:node_index ?pr_n_idx .
#         ?pr_con ns1:layer ?pr_layer .
#         ?pr_layer ns1:layer_index ?pr_l_idx .
#     }
# } GROUP BY ?node ?n_idx ?l_idx
# ORDER BY ?l_idx ?n_idx