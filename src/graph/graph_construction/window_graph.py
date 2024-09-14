import pandas as pd
import networkx as nx
import time
import os


def create_weightless_window_graph(df, src_ip_col, dst_ip_col, window_size=20000, multi_graph=False, line_graph=False, folder_path=None):

    try:
        # Record the start time
        start = time.time()

        graphs = []

        # Total number of records
        total_records = len(df)

        if multi_graph or line_graph:
            base = nx.MultiDiGraph()
        else:
            base = nx.DiGraph()

        # Iterate over the DataFrame in chunks
        i = 0
        for start in range(0, total_records, window_size):
            # Create a chunk of the DataFrame
            df_chunk = df.iloc[start:start + window_size]

            # Create a graph from the chunk
            G = nx.from_pandas_edgelist(df_chunk,
                                        source=src_ip_col,
                                        target=dst_ip_col,
                                        create_using=base)

            if folder_path:
                # Ensure the folder path exists
                os.makedirs(folder_path, exist_ok=True)

                # Define the filename
                filename = os.path.join(folder_path, f'graph_{i}.gexf')

                # Save the graph to a file
                nx.write_gexf(G, filename)

            # Append the graph to the list
            graphs.append(G)
            i += 1

        print(f"Graph created in {time.time() - start:.2f} seconds.")

        return graphs

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-07-01 00:00:01', '2024-07-01 00:00:03', '2024-07-01 00:05:01',
                                     '2024-07-01 00:06:01', '2024-07-01 00:10:01']),
        'src_ip': ['192.168.1.1', '192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.1'],
        'dst_ip': ['192.168.2.1', '192.168.2.1', '192.168.2.1', '192.168.2.2', '192.168.2.1'],
        'src_port': [12345, 12345, 12345, 12346, 12345],
        'dst_port': [80, 80, 80, 80, 80],
        'protocol': ['TCP', 'TCP', 'TCP', 'TCP', 'TCP']
    })

    df2 = create_weightless_window_graph(df, "src_ip", "dst_port", 2)
    print(df2)
