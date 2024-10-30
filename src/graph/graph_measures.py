import networkx as nx
import igraph as ig
import json
import timeit


def calculate_graph_measures(G, file_path=None, verbose=False):

    properties = {}

    start_time = timeit.default_timer()
    number_of_nodes = G.number_of_nodes()
    if verbose:
        print(
            f"==>> number_of_nodes: {number_of_nodes}, in {str(timeit.default_timer() - start_time)} seconds")
    properties["number_of_nodes"] = number_of_nodes

    start_time = timeit.default_timer()
    number_of_edges = G.number_of_edges()
    if verbose:
        print(
            f"==>> number_of_edges: {number_of_edges}, in {str(timeit.default_timer() - start_time)} seconds")
    properties["number_of_edges"] = number_of_edges

    degrees = [degree for _, degree in G.degree()]

    start_time = timeit.default_timer()
    max_degree = max(degrees)
    if verbose:
        print(
            f"==>> max_degree: {max_degree}, in {str(timeit.default_timer() - start_time)} seconds")
    properties["max_degree"] = max_degree

    start_time = timeit.default_timer()
    avg_degree = sum(degrees) / len(degrees)
    if verbose:
        print(
            f"==>> avg_degree: {avg_degree}, in {str(timeit.default_timer() - start_time)} seconds")
    properties["avg_degree"] = avg_degree

    if type(G) == nx.DiGraph or type(G) == nx.Graph:
        start_time = timeit.default_timer()
        transitivity = nx.transitivity(G)
        if verbose:
            print(
                f"==>> transitivity: {transitivity}, in {str(timeit.default_timer() - start_time)} seconds")
        properties["transitivity"] = transitivity

    start_time = timeit.default_timer()
    density = nx.density(G)
    if verbose:
        print(
            f"==>> density: {density}, in {str(timeit.default_timer() - start_time)} seconds")
    properties["density"] = density

    start_time = timeit.default_timer()
    G1 = ig.Graph.from_networkx(G)

<<<<<<< HEAD
    part = G1.community_infomap()
    # part = G1.community_multilevel()
    # part = G1.community_spinglass()
    # part = G1.community_edge_betweenness()
=======
    # part = G1.community_infomap()
    # part = G1.community_label_propagation()
    part = G1.community_leading_eigenvector()
>>>>>>> fa1dce6e8a19e7252b0c8142ec5c25de9cde39f8

    communities = []
    for com in part:
        communities.append([G1.vs[node_index]['_nx_name']
                           for node_index in com])

    # communities = nx.community.louvain_communities(G)
    number_of_communities = len(communities)
    if verbose:
        print(
            f"==>> number_of_communities: {number_of_communities}, in {str(timeit.default_timer() - start_time)} seconds")
    properties["number_of_communities"] = number_of_communities

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

    start_time = timeit.default_timer()
    mixing_parameter = inter_cluster_edges / G.number_of_edges()
    if verbose:
        print(
            f"==>> mixing_parameter: {mixing_parameter}, in {str(timeit.default_timer() - start_time)} seconds")
    properties["mixing_parameter"] = mixing_parameter

    start_time = timeit.default_timer()
    modularity = nx.community.modularity(G, communities)
    if verbose:
        print(
            f"==>> modularity: {modularity}, in {str(timeit.default_timer() - start_time)} seconds")
    properties["modularity"] = modularity

    if file_path:
        outfile = open(file_path, 'w')
        outfile.writelines(json.dumps(properties))
        outfile.close()

    return properties
