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
    """
    A GCN with a dynamic number of layers. 
    ndim_out is a list giving the output dim for each GraphConv layer, e.g. [128, 256, 256].
    """

    def __init__(
        self,
        gcn_in_size,     # dimension of your input features
        # list of output dims per GCN layer, e.g. [128, 256, 256]
        ndim_out,
        gcn_dropout=0.2,
        mlp_hid_size=200,
        n_classes=2,
        mlp_dropout=0.2
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        # The dimension feeding into the first layer
        current_in_dim = gcn_in_size

        # Build each GraphConv layer based on ndim_out
        for out_dim in ndim_out:
            self.convs.append(
                GraphConv(current_in_dim, out_dim, activation=F.relu)
            )
            current_in_dim = out_dim

        # The final output dimension of the GCN stack is the last entry in ndim_out
        final_out_dim = ndim_out[-1]

        # Create MLP Predictor using the final output of the GCN stack
        self.predictor = MLPPredictor(
            final_out_dim,      # in_feats of the MLP
            mlp_hid_size,       # hidden feats of the MLP
            n_classes,          # final number of classes
            dropout=mlp_dropout
        )

        self.dropout = nn.Dropout(gcn_dropout)

    def forward(self, g, features):
        # Optionally add self-loops if needed
        g = dgl.add_self_loop(g)

        h = features
        # Pass through each GCN layer
        for conv in self.convs:
            h = conv(g, h)
            h = self.dropout(h)

        # MLP predictor
        pred = self.predictor(h)
        return pred
