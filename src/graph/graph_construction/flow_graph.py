import pandas as pd
import networkx as nx
import time
import os


def create_weightless_flow_graph(df, src_ip_col, dst_ip_col, multi_graph=False, line_graph=False, new_graph_path=None):

    try:
        # Record the start time
        start = time.time()

        if multi_graph or line_graph:
            base = nx.MultiDiGraph()
        else:
            base = nx.DiGraph()

        # Create the directed graph from the pandas dataframe
        G = nx.from_pandas_edgelist(df,
                                    source=src_ip_col,
                                    target=dst_ip_col,
                                    create_using=base)

        if line_graph:
            G = nx.line_graph(G)

        if new_graph_path:
            # Save the graph to a GEXF file
            nx.write_gexf(G, new_graph_path)

            print(
                f"Graph created and saved to {new_graph_path} in {time.time() - start:.2f} seconds.")

        else:
            print(f"Graph created in {time.time() - start:.2f} seconds.")

        return G
    except Exception as e:
        print(f"An error occurred: {e}")


# def create_weightled_flow_graph(df, src_ip_col, dst_ip_col, weight="none", multi_graph=False, line_graph=False, new_graph_path=None):

#     try:
#         # Record the start time
#         start = time.time()

#         # Create the directed graph from the pandas dataframe
#         G = nx.from_pandas_edgelist(df,
#                                     source=src_ip_col,
#                                     target=dst_ip_col,
#                                     create_using=nx.DiGraph())

#         if weighted == "none":
#             # Calculate total connections for each source IP
#             total_connections = df[src_ip_col].value_counts().to_dict()

#             # Calculate edge weights
#             edge_weights = df.groupby(
#                 [src_ip_col, dst_ip_col]).size().reset_index(name='weight')
#             edge_weights['weight'] = edge_weights.apply(
#                 lambda row: row['weight'] / total_connections[row[src_ip_col]], axis=1)

#         # Add edge weights to the graph
#         for _, row in edge_weights.iterrows():
#             G[row[src_ip_col]][row[dst_ip_col]]['weight'] = row['weight']

#         if
#         # Save the graph to a GEXF file
#         nx.write_gexf(G, new_graph_path)

#         # Print the time taken to create and save the graph
#         end = time.time()
#         print(
#             f"Graph created and saved to {new_graph_path} in {end - start:.2f} seconds.")

#         return G
#     except Exception as e:
#         print(f"An error occurred: {e}")
