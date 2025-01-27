import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, aggregation, num_neighbors=None):
        super(SAGELayer, self).__init__()
        # self.W_msg = nn.Linear(edim, ndim_out)
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
        # print(edges.data['h'].shape)
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}
        # to be experimented
        # return {'m': self.W_msg(edges.data['h'])}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl

            # Neighbor sampling
            if self.num_neighbors:
                sampled_g = dgl.sampling.sample_neighbors(
                    g, g.nodes(), self.num_neighbors)

                # Set node and edge features for the sampled graph
                sampled_g.ndata['h'] = nfeats

                sampled_g.edata['h'] = efeats[sampled_g.edata[dgl.EID]]

                sampled_g.update_all(
                    self.message_func, fn.mean('m', 'h_neigh'))

                h_new = self.activation(self.W_apply(
                    th.cat([nfeats, sampled_g.ndata['h_neigh']], 2)))
            else:

                g.ndata['h'] = nfeats
                g.edata['h'] = efeats

                if self.aggregation == "mean":
                    g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
                elif self.aggregation == "pool":
                    g.update_all(self.message_func, fn.max('m', 'h_pool'))
                    g.ndata['h_neigh'] = self.activation(
                        self.pool_fc(g.ndata['h_pool']))
                elif self.aggregation == "lstm":
                    g.update_all(self.message_func, fn.copy_u('m', 'm_list'))
                    m_list = g.ndata['m_list']
                    batch_size = m_list.shape[0]
                    max_neighbors = m_list.shape[1]
                    lstm_out, _ = self.lstm(m_list.view(
                        batch_size, max_neighbors, -1))
                    # Use the final output of the LSTM
                    g.ndata['h_neigh'] = lstm_out[:, -1, :]

                elif self.aggregation == "gcn":
                    # GCN-style aggregation
                    degs = g.in_degrees().float().clamp(min=1)  # Get in-degree of nodes
                    norm = th.pow(degs, -0.5).unsqueeze(1)      # D^(-1/2)
                    # Scale by D^(-1/2) for source
                    g.ndata['h'] = nfeats * norm
                    # Assign edge features (optional)
                    g.edata['h'] = efeats

                    g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_neigh'))
                    g.ndata['h_neigh'] = g.ndata['h_neigh'] * \
                        norm  # Scale by D^(-1/2) for destination

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
    def __init__(self, ndim_in, edim, ndim_out, num_layers=2, activation=F.relu, aggregation="mean", dropout=0.2, num_neighbors=None, residual=False, num_class=2):
        super().__init__()
        self.gnn = SAGE(ndim_in, edim, ndim_out, num_layers,
                        activation, aggregation, dropout, num_neighbors)
        self.pred = MLPPredictor(
            ndim_out[-1], edim, num_class, activation, residual)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)
