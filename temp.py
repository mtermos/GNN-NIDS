import os
import json


from src.dataset.dataset_info import datasets
from src.numpy_encoder import NumpyEncoder

my_datasets = [
    "cic_ton_iot",
    "cic_ids_2017",
    "cic_bot_iot",
    "cic_ton_iot_modified",
    "ccd_inid_modified",
    "nf_uq_nids_modified",
    "edge_iiot",
    "nf_cse_cic_ids2018",
    "nf_uq_nids",
    "x_iiot",
]
file_name = "df_properties_new.json"
new_file_name = "all_df_properties.json"


data = {
    "cic_ton_iot": {
        "class_diversity_entropy": 0.0001,
        "class_diversity_ratio": 0.5253,
    },
    "cic_ids_2017": {
        "class_diversity_entropy": 0.0001,
        "class_diversity_ratio": 0.3379,
    },
    "cic_bot_iot": {
        "class_diversity_entropy": 0.0329,
        "class_diversity_ratio": 0.5178,
    },
    "cic_ton_iot_modified": {
        "class_diversity_entropy": 0.0001,
        "class_diversity_ratio": 0.4130,
    },
    "ccd_inid_modified": {
        "class_diversity_entropy": 0.2622,
        "class_diversity_ratio": 0.4196,
    },
    "nf_uq_nids_modified": {
        "class_diversity_entropy": 0.1259,
        "class_diversity_ratio": 0.0525,
    },
    "edge_iiot": {
        "class_diversity_entropy": 0.0001,
        "class_diversity_ratio": 0.9998,
    },
    "nf_cse_cic_ids2018": {
        "class_diversity_entropy": 0.0022,
        "class_diversity_ratio": 0.6175,
    },
    "nf_uq_nids": {
        "class_diversity_entropy": 0.0025,
        "class_diversity_ratio": 0.6170,
    },
    "x_iiot": {
        "class_diversity_entropy": 0.0387,
        "class_diversity_ratio": 0.2089,
    },
}

all_props = []
for ds in my_datasets:

    dataset = datasets[ds]
    with open(os.path.join("datasets", dataset.name, file_name), "r") as f:
        prop = json.load(f)
        prop["multi_di_graph"]["class_diversity_entropy"] = data[ds]["class_diversity_entropy"]
        prop["multi_di_graph"]["class_diversity_ratio"] = data[ds]["class_diversity_ratio"]
        all_props.append(prop)

with open(os.path.join(new_file_name), "w") as f:
    f.writelines(json.dumps(all_props, cls=NumpyEncoder))
