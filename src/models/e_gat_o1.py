# Model o1, give it my code and told it to improve it
import torch
import torch.nn as nn
# import torch.nn.functional as F
from dgl.nn.functional import edge_softmax


class EGATLayer(nn.Module):
    """
    Single E-GAT layer using edge features in attention.
    """

    def __init__(self,
                 in_dim_node,
                 in_dim_edge,
                 out_dim,
                 activation=None,
                 dropout=0.0,
                 alpha=0.2,
                 residual=False):
        """
        Parameters
        ----------
        in_dim_node : int
            Dimension of input node features
        in_dim_edge : int
            Dimension of input edge features
        out_dim : int
            Dimension of output node features
        activation : callable or None
            Activation function (e.g., F.relu)
        dropout : float
            Dropout probability
        alpha : float
            Negative slope for LeakyReLU
        residual : bool
            If True, use residual connection
        """
        super(EGATLayer, self).__init__()
        self.in_dim_node = in_dim_node
        self.in_dim_edge = in_dim_edge
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.residual = residual

        # Node transform
        self.W_node = nn.Linear(in_dim_node, out_dim, bias=False)
        # Edge transform
        self.W_edge = nn.Linear(in_dim_edge, out_dim, bias=False)
        # Attention vector (for concatenated [Wh_i, Wh_j, We_ij])
        self.a = nn.Linear(3*out_dim, 1, bias=False)
        # Residual
        if residual and in_dim_node != out_dim:
            self.residual_connection = nn.Linear(
                in_dim_node, out_dim, bias=False)
        else:
            self.residual_connection = None

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def edge_attention(self, edges):
        """
        Compute attention scores given edge data (u->v).
        edges.src['h'] = Wh_i
        edges.dst['h'] = Wh_j
        edges.data['he'] = We_ij
        """
        # Concatenate node features and edge features
        z_concat = torch.cat(
            [edges.src['h'], edges.dst['h'], edges.data['he']], dim=-1)
        # Apply linear layer a, then leaky ReLU
        a_score = self.leaky_relu(self.a(z_concat))
        return {'e': a_score}

    def message_func(self, edges):
        """
        Message = h_i (source node features) * attention coefficient
        """
        # edges.data['alpha'] is the normalized attention
        return {'m': edges.src['h'] * edges.data['alpha']}

    def reduce_func(self, nodes):
        """
        Summation of neighbor messages
        """
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """
        g: DGLGraph
        node_feats: (N, in_dim_node) node features
        edge_feats: (E, in_dim_edge) edge features
        """
        h_in = node_feats  # for residual connection if needed

        # 1) Transform: Wh_i and We_ij
        Wh = self.W_node(node_feats)     # (N, out_dim)
        We = self.W_edge(edge_feats)     # (E, out_dim)

        # Store in the graph
        g.ndata['h'] = Wh
        g.edata['he'] = We

        # 2) Compute unnormalized attention scores
        g.apply_edges(self.edge_attention)

        # 3) Normalize attention scores (softmax over incoming edges)
        g.edata['alpha'] = edge_softmax(g, g.edata['e'])

        # 4) Message passing: multiply node features by alpha
        g.update_all(self.message_func, self.reduce_func)

        # 5) Get the updated node features
        h_out = g.ndata['h']  # (N, out_dim)

        # (Optional) Residual connection
        if self.residual_connection is not None:
            h_out = h_out + self.residual_connection(h_in)
        elif self.residual:
            h_out = h_out + h_in

        # (Optional) Activation
        if self.activation is not None:
            h_out = self.activation(h_out)

        # (Optional) Dropout
        h_out = self.dropout(h_out)

        return h_out


class E_GAT(nn.Module):
    """
    Stack multiple EGATLayer layers
    """

    def __init__(self,
                 in_dim_node,
                 in_dim_edge,
                 hidden_dim,
                 num_layers,
                 activation,
                 dropout,
                 residual=False):
        super(E_GAT, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(EGATLayer(in_dim_node,
                                     in_dim_edge,
                                     hidden_dim,
                                     activation=activation,
                                     dropout=dropout,
                                     residual=residual))

        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(EGATLayer(hidden_dim,
                                         in_dim_edge,
                                         hidden_dim,
                                         activation=activation,
                                         dropout=dropout,
                                         residual=residual))

    def forward(self, g, node_feats, edge_feats):
        h = node_feats
        for layer in self.layers:
            h = layer(g, h, edge_feats)
        return h.sum(dim=1)


# class MLPPredictor(nn.Module):
#     """
#     Simple MLP predictor on top of final node features.
#     """

#     def __init__(self,
#                  in_dim_node,
#                  in_dim_edge,
#                  num_classes,
#                  residual=False):
#         super(MLPPredictor, self).__init__()
#         # Example: a single hidden-layer MLP
#         self.fc1 = nn.Linear(in_dim_node, in_dim_node)
#         self.fc2 = nn.Linear(in_dim_node, num_classes)
#         self.residual = residual

#     def forward(self, g, node_feats):
#         """
#         g : DGLGraph (could be used if you do graph pooling or other readout)
#         node_feats : final node embeddings (N, in_dim_node)
#         """
#         # If you want a graph-level prediction,
#         # you might do something like: h_g = dgl.mean_nodes(g, 'node_feats')
#         # For now, let’s assume node-level classification:
#         x = F.relu(self.fc1(node_feats))
#         x = self.fc2(x)
#         return x

class MLPPredictor(nn.Module):
    """
    An MLP-based edge predictor that scores each edge based on the 
    (src, dst) node representations and optionally the edge's features.

    Parameters
    ----------
    in_features : int
        Dimension of node representation for the final GNN layer.
    e_dim : int
        Dimension of edge features.
    out_classes : int
        Number of output classes or final dimension of scores.
    residual : bool
        Whether to include residual edge features in the final predictor.
    """

    def __init__(self, in_features, e_dim, out_classes, residual=False):
        super().__init__()
        self.residual = residual
        if self.residual:
            # cat([h_u, h_v, h_uv])
            self.fc = nn.Linear(in_features * 2 + e_dim, out_classes)
        else:
            # cat([h_u, h_v])
            self.fc = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        """
        edges.src['h']: shape [num_edges, feat_dim]
        edges.dst['h']: shape [num_edges, feat_dim]
        edges.data['h']: shape [num_edges, 1, e_dim] or [num_edges, e_dim]
        """
        # h_u = edges.src['h'].view(edges.src['h'].shape[0], -1)
        # h_v = edges.dst['h'].view(edges.dst['h'].shape[0], -1)
        h_u = edges.src['h']
        h_v = edges.dst['h']

        if self.residual:
            # If stored as [num_edges, 1, e_dim], flatten the middle dimension
            h_uv = edges.data['he'].view(edges.data['he'].shape[0], -1)
            score = self.fc(torch.cat([h_u, h_v, h_uv], dim=1))
        else:
            score = self.fc(torch.cat([h_u, h_v], dim=1))

        return {'score': score}

    def forward(self, graph, node_feats):
        """
        Predict edge scores for all edges in the graph.

        Parameters
        ----------
        graph : DGLGraph
        node_feats : torch.Tensor
            Shape: [num_nodes, feat_dim].

        Returns
        -------
        torch.Tensor
            Edge scores of shape [num_edges, out_classes].
        """
        with graph.local_scope():
            graph.ndata['h'] = node_feats
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class E_GATModel(nn.Module):
    def __init__(self,
                 ndim_in,    # Node feature dimension
                 edim,       # Edge feature dimension
                 ndim_out,   # Hidden dimension for GAT layers
                 num_layers,
                 activation,
                 dropout,
                 num_class,
                 residual=False):
        # super(E_GATModel, self).__init__()
        super().__init__()
        self.gnn = E_GAT(ndim_in, edim, ndim_out, num_layers,
                         activation, dropout, residual)
        self.pred = MLPPredictor(ndim_out, edim, num_class, residual)

    def forward(self, g, nfeats, efeats):
        # 1. GNN forward pass -> node embeddings
        h = self.gnn(g, nfeats, efeats)
        # 2. MLP head -> predictions
        logits = self.pred(g, h)
        return logits
