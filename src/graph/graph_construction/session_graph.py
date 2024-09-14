import pandas as pd
import networkx as nx
import os
import time


def define_sessions(df, src_ip_col, dst_ip_col, src_port_col, dst_port_col, protocol_col, timeout):
    df = df.sort_values(by='timestamp')
    sessions = []
    current_session_id = 0
    last_seen = {}

    for index, row in df.iterrows():
        five_tuple = (row[src_ip_col], row[dst_ip_col],
                      row[src_port_col], row[dst_port_col], row[protocol_col])
        if five_tuple in last_seen:
            if row['timestamp'] - last_seen[five_tuple] > timeout:
                current_session_id += 1
        else:
            current_session_id += 1
        last_seen[five_tuple] = row['timestamp']
        sessions.append(current_session_id)

    df['session_id'] = sessions
    return df


def create_weightless_session_graph(df, src_ip_col, dst_ip_col, multi_graph=False, line_graph=False, folder_path=None):
    try:
        # Record the start time
        start_time = time.time()

        graphs = []

        if multi_graph or line_graph:
            base_graph_type = nx.MultiDiGraph
        else:
            base_graph_type = nx.DiGraph

        # Iterate over each session in the DataFrame
        for session_id, df_session in df.groupby('session_id'):
            # Create a graph from the session
            G = nx.from_pandas_edgelist(df_session,
                                        source=src_ip_col,
                                        target=dst_ip_col,
                                        create_using=base_graph_type())

            if folder_path:
                # Ensure the folder path exists
                os.makedirs(folder_path, exist_ok=True)

                # Define the filename
                filename = os.path.join(
                    folder_path, f'graph_{session_id}.gexf')

                # Save the graph to a file
                nx.write_gexf(G, filename)

            # Append the graph to the list
            graphs.append(G)

        print(f"Graphs created in {time.time() - start_time:.2f} seconds.")

        return graphs

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-07-01 00:00:01', '2024-07-01 00:00:03', '2024-07-01 00:05:05',
                                     '2024-07-01 00:06:01', '2024-07-01 00:10:01']),
        'src_ip': ['192.168.1.1', '192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.1'],
        'dst_ip': ['192.168.2.1', '192.168.2.1', '192.168.2.1', '192.168.2.2', '192.168.2.1'],
        'src_port': [12345, 12345, 12345, 12346, 12345],
        'dst_port': [80, 80, 80, 80, 80],
        'protocol': ['TCP', 'TCP', 'TCP', 'TCP', 'TCP']
    })

    df2 = define_sessions(df, "src_ip", "dst_port",
                          'src_port', 'dst_port', 'protocol', pd.Timedelta(minutes=5))
    print(df2)
