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

all_props = []
for ds in my_datasets:
    dataset = datasets[ds]
    with open(os.path.join("datasets", dataset.name, file_name), "r") as f:
        prop = json.load(f)
        all_props.append(prop)

with open(os.path.join(new_file_name), "w") as f:
    f.writelines(json.dumps(all_props, cls=NumpyEncoder))
