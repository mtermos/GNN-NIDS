import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation):
        super(GATLayer, self).__init__()
        self.linear = nn.Linear(ndim_in + edim, ndim_out, bias=False)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.attn_fc = nn.Linear(2*ndim_in + edim, 1)
        self.activation = activation
        self.reset_parameters()

    def edge_attention(self, edges):
        return {"e": self.activation(self.attn_fc(th.cat([edges.src["h"], edges.dst["h"], edges.data["h"]], dim=2)))}

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def message_func(self, edges):
        return {"m": self.linear(th.cat([edges.src['h'], edges.data['h']], 2)), "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        z = th.sum(alpha * nodes.mailbox['m'], dim=1)
        # z = th.mean(alpha * nodes.mailbox['m'], dim=1)
        return {'z': z}

    def forward(self, g, nfeats, efeats):
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            g.ndata['h'] = self.activation(self.W_apply(
                th.cat([g.ndata['h'], g.ndata['z']], 2)))
            return g.ndata['h']


class GAT(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(
                    GATLayer(ndim_in, edim, ndim_out[layer], activation))
            else:
                self.layers.append(
                    GATLayer(ndim_out[layer-1], edim, ndim_out[layer], activation))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class MLPPredictor(nn.Module):
    def __init__(self, in_features, edim, out_classes, residual):
        super().__init__()
        self.residual = residual
        if residual:
            self.W = nn.Linear(in_features * 2 + edim, out_classes)
        else:
            self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']

        h_v = edges.dst['h']
        if self.residual:
            h_uv = edges.data['h']
            h_uv = h_uv.view(h_uv.shape[0], h_uv.shape[2])
            score = self.W(th.cat([h_u, h_v, h_uv], 1))
        else:
            score = self.W(th.cat([h_u, h_v], 1))

        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class EGAT(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers=2, activation=F.relu, dropout=0.2, residual=False, num_class=2):
        super().__init__()

        print("e_gat with edge features in attention")

        self.gnn = GAT(ndim_in, edim, ndim_out, num_layers,
                       activation, dropout)

        self.pred = MLPPredictor(ndim_out[-1], edim, num_class, residual)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)
