# mdoel 4o, give it my code and told it to improve it

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class E_GATLayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, dropout=0.2):
        super(E_GATLayer, self).__init__()
        # Linear transformation for node + edge features
        self.node_edge_transform = nn.Linear(
            ndim_in + edim, ndim_out, bias=False)
        # Attention mechanism
        self.attn_fc = nn.Linear(2 * ndim_in + edim, 1)
        # Output feature transformation
        self.out_transform = nn.Linear(
            ndim_in + ndim_out, ndim_out, bias=False)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.node_edge_transform.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.out_transform.weight, gain=gain)

    def edge_attention(self, edges):
        """Compute edge attention scores."""
        z = th.cat([edges.src['h'], edges.dst['h'], edges.data['h']], dim=-1)
        e = self.attn_fc(z)
        return {'e': F.leaky_relu(e)}

    def message_func(self, edges):
        """Compute messages to send to neighbors."""
        m = self.node_edge_transform(
            th.cat([edges.src['h'], edges.data['h']], dim=-1))
        return {'m': m, 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """Aggregate messages from neighbors."""
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        z = th.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'z': z}

    def forward(self, g, nfeats, efeats):
        """Forward pass for the layer."""
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            updated_feats = self.activation(
                self.out_transform(
                    th.cat([g.ndata['h'], g.ndata['z']], dim=-1))
            )
            return self.dropout(updated_feats)


class E_GAT(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, dropout):
        super(E_GAT, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = ndim_in if i == 0 else ndim_out
            self.layers.append(E_GATLayer(
                in_dim, edim, ndim_out, activation, dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, nfeats, efeats):
        for layer in self.layers:
            nfeats = layer(g, nfeats, efeats)
        return nfeats


class MLPPredictor(nn.Module):
    def __init__(self, in_features, edim, out_classes, residual=False):
        super(MLPPredictor, self).__init__()
        self.residual = residual
        input_dim = in_features * 2 + (edim if residual else 0)
        self.fc = nn.Linear(input_dim, out_classes)

    def apply_edges(self, edges):
        """Predict edge scores."""
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        h_edge = edges.data['h'] if self.residual else None
        if self.residual:
            score = self.fc(th.cat([h_src, h_dst, h_edge], dim=-1))
        else:
            score = self.fc(th.cat([h_src, h_dst], dim=-1))
        return {'score': score}

    def forward(self, g, nfeats):
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class E_GATModel(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, dropout, num_classes, residual=False):
        super(E_GATModel, self).__init__()
        self.gnn = E_GAT(ndim_in, edim, ndim_out,
                         num_layers, activation, dropout)
        self.pred = MLPPredictor(ndim_out, edim, num_classes, residual)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)
