import networkx as nx
import igraph as ig
import json
import timeit

import time
from functools import wraps


def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if verbose is in kwargs, defaulting to False if not provided
        verbose = kwargs.get("verbose", False)
        if not verbose:
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            print(
                f"==>> {func.__name__}: {result}, in {str(timeit.default_timer() - start_time)} seconds")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper


@time_execution
def number_of_nodes(G, verbose):
    return G.number_of_nodes()


@time_execution
def number_of_edges(G, verbose):
    return G.number_of_edges()


@time_execution
def transitivity(G, verbose):
    return nx.transitivity(G)


@time_execution
def density(G, verbose):
    return nx.density(G)


@time_execution
def mixing_parameter(G, communities, verbose):

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

    mixing_parameter = inter_cluster_edges / G.number_of_edges()

    return mixing_parameter


@time_execution
def modularity(G, communities, verbose):

    start_time = timeit.default_timer()
    modularity = nx.community.modularity(G, communities)
    if verbose:
        print(
            f"==>> modularity: {modularity}, in {str(timeit.default_timer() - start_time)} seconds")

    return modularity


def get_degrees(G, verbose):
    start_time = timeit.default_timer()
    degrees = [degree for _, degree in G.degree()]
    if verbose:
        print(
            f"==>> calculated degrees, in {str(timeit.default_timer() - start_time)} seconds")
    return degrees


def find_communities(G, verbose):

    start_time = timeit.default_timer()
    G1 = ig.Graph.from_networkx(G)

    part = G1.community_infomap()
    # part = G1.community_multilevel()
    # part = G1.community_spinglass()
    # part = G1.community_edge_betweenness()

    communities = []
    for com in part:
        communities.append([G1.vs[node_index]['_nx_name']
                           for node_index in com])

    # communities = nx.community.louvain_communities(G)
    if verbose:
        print(
            f"==>> number_of_communities: {len(communities)}, in {str(timeit.default_timer() - start_time)} seconds")

    return communities


def calculate_graph_measures(G, file_path=None, verbose=False):

    properties = {}

    properties["number_of_nodes"] = number_of_nodes(G, verbose)
    properties["number_of_edges"] = number_of_edges(G, verbose)

    degrees = get_degrees(G, verbose)

    properties["max_degree"] = max(degrees)
    properties["avg_degree"] = sum(degrees) / len(degrees)

    if type(G) == nx.DiGraph or type(G) == nx.Graph:
        properties["transitivity"] = transitivity(G, verbose)

    properties["density"] = density(G, verbose)

    # communities = find_communities(G, verbose)

    # properties["number_of_communities"] = len(communities)
    # properties["mixing_parameter"] = mixing_parameter(G, communities, verbose)
    # properties["modularity"] = modularity(G, communities, verbose)

    if file_path:
        outfile = open(file_path, 'w')
        outfile.writelines(json.dumps(properties))
        outfile.close()

    return properties
