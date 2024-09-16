import networkx as nx
import igraph as ig
import json


def calculate_graph_measures(G, file_path=None):

    properties = {}

    properties["number_of_nodes"] = G.number_of_nodes()
    properties["number_of_edges"] = G.number_of_edges()

    degrees = [degree for _, degree in G.degree()]
    properties["max_degree"] = max(degrees)
    properties["avg_degree"] = sum(degrees) / len(degrees)

    if type(G) == nx.DiGraph or type(G) == nx.Graph:
        properties["transitivity"] = nx.transitivity(G)

    properties["density"] = nx.density(G)

    G1 = ig.Graph.from_networkx(G)
    part = G1.community_infomap()

    communities = []
    for com in part:
        communities.append([G1.vs[node_index]['_nx_name']
                           for node_index in com])

    properties["number_of_communities"] = len(communities)

    # Step 1: Map each node to its community
    node_to_community = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_index

    # Step 2: Count inter-cluster edges efficiently
    inter_cluster_edges = 0
    for u, v in G.edges():
        # Directly check if u and v belong to different communities
        if node_to_community[u] != node_to_community[v]:
            inter_cluster_edges += 1

    properties["mixing_parameter"] = inter_cluster_edges / G.number_of_edges()

    properties["modularity"] = nx.community.modularity(G, communities)

    if file_path:
        outfile = open(file_path, 'w')
        outfile.writelines(json.dumps(properties))
        outfile.close()

    return properties
