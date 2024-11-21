import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl


class GCNLayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation):
        super(GCNLayer, self).__init__()
        # force to outut fix dimensions
        self.W_msg = nn.Linear(ndim_in + edim, ndim_out)
        # apply weight
        self.W_apply = nn.Linear(ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        combined = th.cat([edges.src['h'], edges.data['h']], dim=-1)
        message = self.W_msg(combined)
        norm_weight = edges.data['norm_weight'].unsqueeze(-1).unsqueeze(-1)
        message = norm_weight * message
        return {'m': message}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl

            g.ndata['h'] = nfeats
            g.edata['h'] = efeats

            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

            # g.ndata['h'] = F.relu(self.W_apply(
            #     th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))

            # g.ndata['h_neigh'] = g.ndata['h_neigh'] * \
            #     g.ndata['norm'].unsqueeze(-1)

            g.ndata['h'] = F.relu(self.W_apply(g.ndata['h_neigh']))
            return g.ndata['h']


class GCN(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(ndim_in, edim, 128, activation))
        self.layers.append(GCNLayer(128, edim, ndim_out, activation))
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


class EGCN(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, dropout, residual):
        super().__init__()
        self.gnn = GCN(ndim_in, edim, ndim_out, activation, dropout)
        self.pred = MLPPredictor(ndim_out, edim, 2, residual)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)
