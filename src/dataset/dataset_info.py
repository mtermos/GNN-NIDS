class DatasetInfo:
    def __init__(
            self,
            name,
            path,
            file_type,

            # Key Columns names
            src_ip_col,
            src_port_col,
            dst_ip_col,
            dst_port_col,
            flow_id_col,
            timestamp_col,
            label_col,
            class_col,

            class_num_col=None,
            timestamp_format=None,

            # Columns to be dropped from the dataset during preprocessing.
            drop_columns=[],

            # Columns to be dropped from the dataset during preprocessing.
            weak_columns=[],
    ):

        self.name = name
        self.path = path
        self.file_type = file_type
        self.src_ip_col = src_ip_col
        self.src_port_col = src_port_col
        self.dst_ip_col = dst_ip_col
        self.dst_port_col = dst_port_col
        self.flow_id_col = flow_id_col
        self.timestamp_col = timestamp_col
        self.timestamp_format = timestamp_format
        self.label_col = label_col
        self.class_col = class_col
        self.class_num_col = class_num_col
        self.drop_columns = drop_columns
        self.weak_columns = weak_columns


datasets_list = [
    DatasetInfo(name="cic_ton_iot_5_percent",
                path="datasets/cic_ton_iot_5_percent/cic_ton_iot_5_percent.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="%d/%m/%Y %I:%M:%S %p",

                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Subflow Bwd Pkts', 'Flow IAT Mean', 'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Bwd Header Len', 'Tot Bwd Pkts', 'Bwd Pkt Len Mean', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
                              'CWE Flag Count', 'Bwd IAT Tot', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Min', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max', 'Pkt Len Var', 'FIN Flag Cnt', 'Bwd IAT Mean', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Tot', 'PSH Flag Cnt', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="cic_ton_iot",
                path="datasets/cic_ton_iot/cic_ton_iot.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="%d/%m/%Y %I:%M:%S %p",

                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Subflow Bwd Pkts', 'Active Mean', 'Active Std', 'Bwd Header Len', 'Bwd IAT Mean', 'Bwd IAT Tot', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Bwd Seg Size Avg', 'CWE Flag Count', 'FIN Flag Cnt',
                              'Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Min', 'Flow Pkts/s', 'Fwd IAT Mean', 'Fwd IAT Tot', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Fwd Seg Size Avg', 'Idle Mean', 'PSH Flag Cnt', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'Pkt Size Avg', 'Tot Bwd Pkts', 'TotLen Bwd Pkts']
                ),
    DatasetInfo(name="cic_ids_2017_5_percent",
                path="datasets/cic_ids_2017_5_percent/cic_ids_2017_5_percent.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Fwd IAT Min',  'Idle Max', 'Flow IAT Mean',  'Protocol',   'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Subflow Fwd Pkts', 'Bwd Pkt Len Mean', 'Tot Bwd Pkts', 'Pkt Size Avg',
                              'Subflow Bwd Pkts', 'Bwd IAT Std', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Std', 'Fwd URG Flags', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max',  'Pkt Len Var',  'Tot Fwd Pkts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd PSH Flags', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Max',  'Fwd IAT Tot', 'RST Flag Cnt', 'Subflow Bwd Byts', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="cic_ids_2017",
                path="./datasets/cic_ids_2017.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Fwd IAT Min',  'Idle Max', 'Flow IAT Mean',  'Protocol',   'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Subflow Fwd Pkts', 'Bwd Pkt Len Mean', 'Tot Bwd Pkts', 'Pkt Size Avg',
                              'Subflow Bwd Pkts', 'Bwd IAT Std', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Std', 'Fwd URG Flags', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max',  'Pkt Len Var',  'Tot Fwd Pkts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd PSH Flags', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Max',  'Fwd IAT Tot', 'RST Flag Cnt', 'Subflow Bwd Byts', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="nf_bot_iot",
                path="./datasets/nf_bot_iot.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="Dst IP",
                dst_port_col="L4_DST_PORT",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["L4_SRC_PORT", "L4_DST_PORT", "Attack"],
                ),
    DatasetInfo(name="edge_iiot",
                path="./datasets/edge_iiot.parquet",
                file_type="parquet",
                src_ip_col="ip.src_host",
                src_port_col="Src Port",
                dst_ip_col="ip.dst_host",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Attack_label",
                class_col="Attack_type",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["ip.src_host", "ip.dst_host", "tcp.options", "tcp.payload", "mqtt.msg,icmp.unused",
                              "http.tls_port", "dns.qry.type", "mqtt.msg_decoded_as", "frame.time", "Attack_type"]
                ),
    DatasetInfo(name="nf_cse_cic_ids2018",
                path="./datasets/nf_cse_cic_ids2018.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="Dst IP",
                dst_port_col="L4_DST_PORT",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["L4_SRC_PORT", "L4_DST_PORT", "Attack"],
                ),
    DatasetInfo(name="nf_bot_iotv2",
                path="./datasets/nf_cse_cic_ids2018.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="Dst IP",
                dst_port_col="L4_DST_PORT",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["L4_SRC_PORT", "L4_DST_PORT", "Attack"],
                ),
    DatasetInfo(name="nf_uq_nids",
                path="./datasets/nf_uq_nids.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="Dst IP",
                dst_port_col="L4_DST_PORT",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["L4_SRC_PORT",
                              "L4_DST_PORT", "Attack", "Dataset"],
                ),
    DatasetInfo(name="x_iiot",
                path="./datasets/x_iiot.parquet",
                file_type="parquet",
                src_ip_col="Scr_IP",
                src_port_col="Scr_port",
                dst_ip_col="Des_IP",
                dst_port_col="Des_port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="class3",
                class_col="class2",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Scr_IP", "Scr_port", "Des_IP",
                              "Des_port", "Timestamp", "Date", "class1", "class2"],
                ),
    DatasetInfo(name="cic_ton_iot_modified",
                path="./datasets/cic_ton_iot_modified.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP", "Timestamp",
                              "Src Port", "Dst Port", "Attack", "datetime"],
                ),
    DatasetInfo(name="nf_ton_iotv2_modified",
                path="./datasets/nf_ton_iotv2_modified.parquet",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "Attack", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
                              "FLOW_DURATION_MILLISECONDS", "LONGEST_FLOW_PKT", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "TCP_WIN_MAX_IN TCP_WIN_MAX_OUT"],
                ),
    DatasetInfo(name="ccd_inid_modified",
                path="./datasets/ccd_inid_modified.parquet",
                file_type="parquet",
                src_ip_col="src_ip",
                src_port_col="src_port",
                dst_ip_col="dst_ip",
                dst_port_col="dst_port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="traffic_type",
                class_col="atk_type",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["id", "src_ip", "src_port",
                              "dst_ip", "dst_port", "atk_type"],
                ),
    DatasetInfo(name="nf_uq_nids_modified",
                path="./datasets/nf_uq_nids_modified.parquet",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack", "Dataset"],
                ),
]


datasets = {dataset.name: dataset for dataset in datasets_list}
