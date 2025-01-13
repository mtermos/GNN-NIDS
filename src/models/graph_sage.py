import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import SAGEConv


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


class GRAPHSAGE(nn.Module):
    def __init__(
        self,
        in_dim,              # input feature dimension
        # list of output dimensions for each layer, e.g. [128, 256, 256]
        ndim_out,
        aggregator_type="gcn",
        sage_dropout=0.2,
        mlp_hid_size=200,
        n_classes=2,
        mlp_dropout=0.2
    ):
        super().__init__()

        # A ModuleList to store SAGEConv layers
        self.layers = nn.ModuleList()

        # The dimension of the features entering the first layer is in_dim
        current_in_dim = in_dim

        # Create each layer based on ndim_out
        for out_dim in ndim_out:
            self.layers.append(
                SAGEConv(
                    in_feats=current_in_dim,
                    out_feats=out_dim,
                    aggregator_type=aggregator_type,
                    activation=F.relu
                )
            )
            # The output of this layer becomes the input to the next
            current_in_dim = out_dim

        # The final layer's dimension is the last entry in ndim_out
        final_out_dim = ndim_out[-1]

        # Create MLP Predictor with final_out_dim as input
        self.predictor = MLPPredictor(
            final_out_dim, mlp_hid_size, n_classes, dropout=mlp_dropout
        )

        self.dropout = nn.Dropout(sage_dropout)

    def forward(self, g, features):
        # Optionally add self-loops if needed
        g = dgl.add_self_loop(g)

        h = features
        # Pass through each SAGEConv layer
        for layer in self.layers:
            h = layer(g, h)
            h = self.dropout(h)

        # Predict
        pred = self.predictor(h)
        return pred
