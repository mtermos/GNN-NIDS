import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv


class MLPPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats, output, dropout=0.):
        super(MLPPredictor, self).__init__()

        if output == 1:
            self.predict = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats),
                nn.Linear(hidden_feats, output),
                nn.Sigmoid()
            )
        else:
            self.predict = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats),
                nn.Linear(hidden_feats, output),
                nn.Softmax()
            )

    def forward(self, feats):
        return self.predict(feats)


class GAT(nn.Module):
    def __init__(self,
                 gcn_in_size,
                 gcn_hid_size=128,
                 gcn_out_size=128,
                 gcn_dropout=0.2,
                 num_heads=4,
                 mlp_hid_size=200,
                 n_classes=2,
                 mlp_dropout=0.2):

        super().__init__()
        self.conv1 = GATConv(gcn_in_size, gcn_hid_size,
                             num_heads=num_heads, activation=F.relu)
        self.conv2 = GATConv(gcn_hid_size, gcn_out_size,
                             num_heads=num_heads, activation=F.relu)

        if n_classes == 2:
            self.predictor = MLPPredictor(
                gcn_out_size, mlp_hid_size, 1, dropout=mlp_dropout)
        else:
            self.predictor = MLPPredictor(
                gcn_out_size, mlp_hid_size, n_classes, dropout=mlp_dropout)

        self.dropout = nn.Dropout(gcn_dropout)

    def forward(self, g, features):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, features)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = self.dropout(h)
        pred = self.predictor(h)
        return pred
