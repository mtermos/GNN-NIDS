import pandas as pd
import os

import networkx as nx
from src.dataset.dataset_info import datasets

name = "cic_ids_2017_5_percent"

dataset = datasets[name]
dataset_folder_path = os.path.join(
    "datasets", name, "session_graphs", "graphs")

graphs = []
for file in os.listdir(dataset_folder_path):
    G = nx.read_gexf(file)
    break

G
