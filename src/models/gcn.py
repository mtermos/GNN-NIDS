import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv


class MLPPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats, output, dropout=0.):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, output)
        )

    def forward(self, feats):
        return self.predict(feats)


class GCN(nn.Module):
    def __init__(self,
                 gcn_in_size,
                 gcn_hid_size=128,
                 gcn_out_size=128,
                 gcn_dropout=0.2,
                 mlp_hid_size=200,
                 n_classes=2,
                 mlp_dropout=0.2):

        super().__init__()
        self.conv1 = GraphConv(gcn_in_size, gcn_hid_size, activation=F.relu)
        self.conv2 = GraphConv(gcn_hid_size, gcn_out_size, activation=F.relu)

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
