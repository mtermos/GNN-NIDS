import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv


class MLPPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, output, dropout=0.):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, output),
        )

    def forward(self, feats):
        return self.predict(feats)


class GAT(nn.Module):
    """
    GAT with a dynamic number of layers. 
    'ndim_out' is a list of output dimensions for each GAT layer 
    (e.g., [128, 256, 256] for three layers).
    """

    def __init__(
        self,
        in_dim,           # dimension of your input node features
        ndim_out,         # list of output dims, e.g. [128, 256, 256]
        num_heads=4,
        dropout=0.2,
        mlp_hid_size=200,
        n_classes=2,
        mlp_dropout=0.2
    ):
        super().__init__()

        # We'll store the GATConv layers in a ModuleList
        self.layers = nn.ModuleList()

        # Track the "current" dimensionality of features going into the next layer
        current_in_dim = in_dim

        for out_dim in ndim_out:
            self.layers.append(
                GATConv(
                    in_feats=current_in_dim,
                    out_feats=out_dim,
                    num_heads=num_heads,
                    activation=F.relu
                )
            )
            # After averaging over heads, the new feature dim = out_dim
            current_in_dim = out_dim

        # The last dimension in ndim_out is what feeds into the MLP
        final_out_dim = ndim_out[-1]

        # Create an MLP predictor
        self.predictor = MLPPredictor(
            in_feats=final_out_dim,
            hidden_feats=mlp_hid_size,
            output=n_classes,
            dropout=mlp_dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, g, features):
        # Optionally add self-loop if needed (depends on whether your graph already has them)
        g = dgl.add_self_loop(g)

        h = features
        for gat_layer in self.layers:
            # Each GAT layer outputs shape (N, num_heads, out_dim)
            h = gat_layer(g, h)
            # Aggregate over the heads by mean or any other aggregator you prefer
            h = h.mean(dim=1)   # shape (N, out_dim)
            h = self.dropout(h)

        # Final MLP prediction
        pred = self.predictor(h)  # shape (N, n_classes)
        return pred
