import rdflib
from rdflib.namespace import RDF, RDFS
import src

RDFS
NNC = rdflib.Namespace("http://example.org/")   # Neural Network Connections (I guess I should have a location that is not example.org?)

if __name__ == "__main__":
    config = src.cfg.ConfigObject.get_param_from_args()
    config("PruningSelection", "Reset")
    load = src.standardLoad(existing_config=config, index=0)
    model = src.modelstruct.getModel(load["config"])
    layer = None
    last_layer = []
    this_layer = []

    g = rdflib.Graph()
    g.bind("ns1", NNC)

    count = 0

    # Define the input vector layer of the network
    input_layer = rdflib.BNode()
    g.add((input_layer, RDF.type, NNC.layer))
    g.add((input_layer, RDFS.label, rdflib.Literal("Input")))
    g.add((input_layer, NNC.layer_index, rdflib.Literal(-1)))

    for mod_name, module in model.named_modules():
        if "fc" in mod_name:
            last = layer if layer is not None else rdflib.BNode()
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
                    for inputNum, _ in enumerate(node):
                        i = rdflib.BNode()
                        g.add((i, RDF.type, NNC.node))
                        g.add((i, RDF.type, NNC.input_node))
                        g.add((i, NNC.layer, input_layer))
                        g.add((i, NNC.node_index, rdflib.Literal(inputNum)))
                        last_layer.append(i)
                    g.add((input_layer, NNC.num_nodes, rdflib.Literal(len(node))))

                max_connection = None
                for connected_to, connec in enumerate(node):
                    if abs(connec.item()) > 0.001:
                        c = rdflib.BNode()
                        g.add((c, RDF.type, NNC.connection))
                        g.add((c, NNC.weight, rdflib.Literal(connec.item())))
                        g.add((c, NNC.node, n))

                        if len(last_layer) > connected_to:
                            g.add((c, NNC.connected_to, last_layer[connected_to]))

                        if max_connection is None or max_connection[0] < abs(connec.item()):
                            max_connection = (abs(connec.item()), connected_to)

                if max_connection and len(last_layer) > max_connection[1]:
                    g.add((n, NNC.primary_contributor, last_layer[max_connection[1]]))

            last_layer = this_layer
            this_layer = []
            count += 1

    g.serialize("datasets/model.ttl", encoding="UTF-8")


# PREFIX ns1: <http://example.org/>
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