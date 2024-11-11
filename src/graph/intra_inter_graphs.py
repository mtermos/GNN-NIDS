import networkx as nx


def separate_graph(graph, communities):
    """
    Separates a graph into intra-community and inter-community edges.

    Parameters:
    - graph: A NetworkX graph
    - communities: A list of sets, where each set contains the nodes in a community

    Returns:
    - intra_graph: A graph containing only intra-community edges
    - inter_graph: A graph containing only inter-community edges
    """
    # Create new graphs for intra-community and inter-community edges
    intra_graph = nx.Graph()
    inter_graph = nx.Graph()

    # Add all nodes to both graphs to ensure structure is maintained
    intra_graph.add_nodes_from(graph.nodes())
    inter_graph.add_nodes_from(graph.nodes())

    # Organize communities in a way that allows quick lookup of node to community mapping
    node_to_community = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_index

    # Iterate through each edge in the original graph
    for edge in graph.edges():
        node_u, node_v = edge

        # Determine if an edge is intra-community or inter-community
        if node_to_community.get(node_u) == node_to_community.get(node_v):
            # Intra-community edge
            intra_graph.add_edge(node_u, node_v)
        else:
            # Inter-community edge
            inter_graph.add_edge(node_u, node_v)

    return intra_graph, inter_graph
