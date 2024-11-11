import networkx as nx
import igraph as ig


def _rescale(betweenness, n, normalized, directed=False, k=None, endpoints=False):
    if normalized:
        if endpoints:
            if n < 2:
                scale = None  # no normalization
            else:
                # Scale factor should include endpoint nodes
                scale = 1 / (n * (n - 1))
        elif n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def cal_betweenness_centrality(G):
    G1 = ig.Graph.from_networkx(G)
    estimate = G1.betweenness(directed=True)
    b = dict(zip(G1.vs["_nx_name"], estimate))
    return _rescale(b, G1.vcount(), True)
