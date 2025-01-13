# Model o1, give it my code and told it to improve it

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GATLayer(nn.Module):
    """
    A single Graph Attention (GAT) Layer that incorporates edge features.

    Parameters
    ----------
    in_dim : int
        Dimension of node features.
    e_dim : int
        Dimension of edge features.
    out_dim : int
        Dimension of the output node features.
    activation : callable
        Non-linear activation function.
    """

    def __init__(self, in_dim, e_dim, out_dim, activation=F.relu, negative_slope=0.2):
        super().__init__()

        # Linear transformations
        # 1) For incorporating edge features into node features
        self.project_edge = nn.Linear(in_dim + e_dim, out_dim, bias=False)

        # 2) Final node update after aggregation
        self.W_apply = nn.Linear(in_dim + out_dim, out_dim, bias=True)

        # Attention mechanism: we only need to compare source/destination node features
        #   shape after concat: [N_edges, 1, 2*in_dim]
        #   the output is a scalar (or 1-dim) per edge
        self.attn_fc = nn.Linear(2 * in_dim, 1, bias=True)

        # Activation function
        self.activation = activation

        # LeakyReLU negative slope
        self.negative_slope = negative_slope

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters using Xavier initialization."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.project_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.W_apply.weight, gain=gain)
        if self.attn_fc.bias is not None:
            nn.init.zeros_(self.attn_fc.bias)
        if self.W_apply.bias is not None:
            nn.init.zeros_(self.W_apply.bias)

    def edge_attention(self, edges):
        """
        Compute unnormalized attention scores on edges.
        edges.src['h']: [num_edges, 1, in_dim]
        edges.dst['h']: [num_edges, 1, in_dim]
        """
        # Concatenate the source and destination node features
        concat_feat = torch.cat([edges.src["h"], edges.dst["h"]], dim=2)
        # Calculate attention score
        e = self.attn_fc(concat_feat)
        return {"e": F.leaky_relu(e, negative_slope=self.negative_slope)}

    def message_func(self, edges):
        """
        Prepare messages to send to destination nodes. 
        This includes:
          - projected features combining source node features + edge features
          - attention scores "e"
        """
        # edges.src['h']: [num_edges, 1, in_dim]
        # edges.data['h']: [num_edges, 1, e_dim]
        # -> cat => [num_edges, 1, (in_dim + e_dim)]
        edge_cat = torch.cat([edges.src['h'], edges.data['h']], dim=2)
        # project_edge -> [num_edges, 1, out_dim]
        m = self.project_edge(edge_cat)
        return {
            "m": m,
            "e": edges.data["e"]
        }

    def reduce_func(self, nodes):
        """
        Aggregate incoming messages using attention scores.

        nodes.mailbox['m']: [num_nodes, num_incoming_edges, 1, out_dim]
        nodes.mailbox['e']: [num_nodes, num_incoming_edges, 1, 1]
        """
        # Apply softmax over the attention scores along incoming edges
        # shape: [num_nodes, num_incoming_edges, 1, 1]
        alpha = F.softmax(nodes.mailbox['e'], dim=1)

        # Weighted sum of incoming messages
        # alpha * nodes.mailbox['m'] => [num_nodes, num_incoming_edges, 1, out_dim]
        # Summation over the 2nd dimension (num_incoming_edges)
        # shape: [num_nodes, 1, out_dim]
        z = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'z': z}

    def forward(self, g, nfeats, efeats):
        """
        Forward pass for GATLayer.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        nfeats : torch.Tensor
            Node features with shape [num_nodes, 1, in_dim] (or [num_nodes, in_dim] if not batching heads).
        efeats : torch.Tensor
            Edge features with shape [num_edges, 1, e_dim] (or [num_edges, e_dim] if not batching heads).

        Returns
        -------
        torch.Tensor
            Updated node features of shape [num_nodes, 1, out_dim].
        """
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats

            # Compute unnormalized attention scores
            g.apply_edges(self.edge_attention)

            # Message passing + aggregation
            g.update_all(self.message_func, self.reduce_func)

            # Combine original node features with the aggregated result
            updated_node_feats = torch.cat([g.ndata['h'], g.ndata['z']], dim=2)
            out = self.activation(self.W_apply(updated_node_feats))

            return out


class GAT(nn.Module):
    """
    Multi-layer GAT model that stacks multiple GATLayer layers.

    Parameters
    ----------
    in_dim : int
        Input dimension for the node features of the first layer.
    e_dim : int
        Edge feature dimension, used throughout all layers.
    out_dims : list of int
        List of output dimensions for each GATLayer. E.g., [64, 32].
    num_layers : int
        Number of GAT layers.
    activation : callable
        Non-linear activation function.
    dropout : float
        Dropout probability.
    """

    def __init__(self, in_dim, e_dim, out_dims, num_layers, activation=F.relu, dropout=0.2):
        super().__init__()
        assert len(out_dims) == num_layers, \
            "Length of out_dims must match num_layers."

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.layers.append(
                    GATLayer(in_dim, e_dim, out_dims[layer_idx], activation))
            else:
                self.layers.append(
                    GATLayer(out_dims[layer_idx - 1], e_dim, out_dims[layer_idx], activation))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        """
        Forward pass through the stacked GAT layers.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        nfeats : torch.Tensor
            Node features, shape [num_nodes, 1, in_dim].
        efeats : torch.Tensor
            Edge features, shape [num_edges, 1, e_dim].

        Returns
        -------
        torch.Tensor
            The final node representation from the last GAT layer.
            Shape: [num_nodes, 1, out_dims[-1]].
        """
        h = nfeats
        for i, layer in enumerate(self.layers):
            # Apply dropout to the intermediate representations
            if i > 0:
                h = self.dropout(h)
            h = layer(g, h, efeats)

        # If you only have one head, sum(1) might simply remove the middle dimension:
        # e.g. shape = [num_nodes, 1, feat_dim] -> [num_nodes, feat_dim]
        # Adjust as necessary if you keep that extra dimension for your use case.
        return h.sum(dim=1)


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
        h_u = edges.src['h']
        h_v = edges.dst['h']

        if self.residual:
            # If stored as [num_edges, 1, e_dim], flatten the middle dimension
            h_uv = edges.data['h'].view(edges.data['h'].shape[0], -1)
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


class EGAT(nn.Module):
    """
    EGAT model that combines:
      1) A multi-layer GAT architecture with edge features
      2) An MLP-based edge predictor on top of the final node features

    Parameters
    ----------
    in_dim : int
        Input dimension of node features.
    e_dim : int
        Input dimension of edge features.
    out_dims : list of int
        Output dimensions for each GAT layer.
    num_layers : int
        Number of GAT layers.
    activation : callable
        Activation function used in GAT layers.
    dropout : float
        Dropout probability in GAT layers.
    residual : bool
        Whether to use residual edge features in the predictor.
    num_classes : int
        Number of classes for the final prediction.
    """

    def __init__(self,
                 in_dim,
                 e_dim,
                 out_dims,
                 num_layers=2,
                 activation=F.relu,
                 dropout=0.2,
                 residual=False,
                 num_classes=2):
        super().__init__()

        # GAT backbone
        self.gnn = GAT(in_dim, e_dim, out_dims,
                       num_layers, activation, dropout)

        # Edge predictor
        self.pred = MLPPredictor(out_dims[-1], e_dim, num_classes, residual)

    def forward(self, g, nfeats, efeats):
        """
        Forward pass: 
          1) GNN to get node representations
          2) MLP predictor for edge scores

        Parameters
        ----------
        g : DGLGraph
        nfeats : torch.Tensor
            Node features [num_nodes, 1, in_dim] or [num_nodes, in_dim].
        efeats : torch.Tensor
            Edge features [num_edges, 1, e_dim] or [num_edges, e_dim].

        Returns
        -------
        torch.Tensor
            Edge classification or regression scores of shape [num_edges, num_classes].
        """
        # 1) Obtain node embeddings
        node_embeddings = self.gnn(g, nfeats, efeats)

        # 2) Predict edge scores
        scores = self.pred(g, node_embeddings)
        return scores
