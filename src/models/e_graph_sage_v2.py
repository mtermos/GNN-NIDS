import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, aggregation, num_neighbors=None):
        super(SAGELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edim, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation
        self.aggregation = aggregation
        self.num_neighbors = num_neighbors

        if aggregation == "pool":
            self.pool_fc = nn.Linear(ndim_out, ndim_out)
        elif aggregation == "lstm":
            self.lstm = nn.LSTM(ndim_out, ndim_out, batch_first=True)

    def message_func(self, edges):
        # if multi_graph then the node features of the source node are repeated
        # after concatenation, for each edge, we have [src_nfeats_1 , ... , src_nfeats_n, efeats_1, ... efeats_m]
        # after that we apply linear layer to create new featurescset called m.
        # return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}
        return {'m': edges.data['h']}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl

            # Neighbor sampling
            if self.num_neighbors:
                g = dgl.sampling.sample_neighbors(
                    g, g.nodes(), self.num_neighbors)

                g.ndata['h'] = nfeats
                g.edata['h'] = efeats[g.edata[dgl.EID]]

            else:

                g.ndata['h'] = nfeats
                g.edata['h'] = efeats

            if self.aggregation == "mean":
                g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            elif self.aggregation == "pool":
                g.update_all(self.message_func, fn.max('m', 'h_pool'))
                g.ndata['h_neigh'] = self.activation(
                    self.pool_fc(g.ndata['h_pool']))
            h_new = self.activation(self.W_apply(
                th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return h_new


class SAGE(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, aggregation, dropout, num_neighbors):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(
                    SAGELayer(ndim_in, edim, ndim_out[layer], activation, aggregation, num_neighbors[layer] if num_neighbors else None))
            else:
                self.layers.append(SAGELayer(
                    ndim_out[layer-1], edim, ndim_out[layer], activation, aggregation, num_neighbors[layer] if num_neighbors else None))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):

        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class MLPPredictor(nn.Module):
    def __init__(self, in_features, edim, out_classes, activation, residual):
        super().__init__()
        self.activation = activation
        self.residual = residual
        if residual:
            self.W = nn.Linear(in_features * 2 + edim, 100)
        else:
            self.W = nn.Linear(in_features * 2, 100)
        self.linear = nn.Linear(100, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        if self.residual:
            h_uv = edges.data['h']
            h_uv = h_uv.view(h_uv.shape[0], h_uv.shape[2])
            score = self.W(th.cat([h_u, h_v, h_uv], 1))
        else:
            score = self.W(th.cat([h_u, h_v], 1))
        score = self.linear(score)
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class EGRAPHSAGE(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers=2, activation=F.relu, aggregation="mean", dropout=0.2, num_neighbors=None, residual=True, num_class=2):
        super().__init__()
        self.gnn = SAGE(ndim_in, edim, ndim_out, num_layers,
                        activation, aggregation, dropout, num_neighbors)
        self.pred = MLPPredictor(
            ndim_out[-1], edim, num_class, activation, residual)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)
