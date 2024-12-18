import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, num_neighbors=None):
        super(SAGELayer, self).__init__()
        # force to outut fix dimensions
        self.W_msg = nn.Linear(ndim_in + edim, ndim_out)
        # apply weight
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation
        self.num_neighbors = num_neighbors

    def message_func(self, edges):
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl

            # Neighbor sampling
            if self.num_neighbors:
                sampled_g = dgl.sampling.sample_neighbors(
                    g, g.nodes(), self.num_neighbors)

                # Set node and edge features for the sampled graph
                sampled_g.ndata['h'] = nfeats
                # print(nfeats.shape)

                sampled_eids = sampled_g.edata[dgl.EID]

                # Create new edge features for the sampled graph
                sampled_eids = sampled_g.edata[dgl.EID]
                sampled_g.edata['h'] = efeats[sampled_eids]

                sampled_g.update_all(
                    self.message_func, fn.mean('m', 'h_neigh'))

                h_new = F.relu(self.W_apply(
                    th.cat([nfeats, sampled_g.ndata['h_neigh']], 2)))
            else:

                g.ndata['h'] = nfeats
                g.edata['h'] = efeats

                g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

                h_new = F.relu(self.W_apply(
                    th.cat([nfeats, g.ndata['h_neigh']], 2)))
            return h_new


class SAGE(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, dropout, num_neighbors):
        super(SAGE, self).__init__()
        if num_neighbors:
            self.conv1 = SAGELayer(
                ndim_in, edim, 128, activation, num_neighbors[0])
            self.conv2 = SAGELayer(128, edim, ndim_out,
                                   activation, num_neighbors[1])
        else:
            self.conv1 = SAGELayer(ndim_in, edim, 128, activation)
            self.conv2 = SAGELayer(128, edim, ndim_out,
                                   activation)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        nfeats = self.conv1(g, nfeats, efeats)
        nfeats = self.dropout(nfeats)
        nfeats = self.conv2(g, nfeats, efeats)

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


class EGRAPHSAGE(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, dropout, num_neighbors=None, residual=False, num_class=2):
        super().__init__()
        self.gnn = SAGE(ndim_in, edim, ndim_out,
                        activation, dropout, num_neighbors)
        self.pred = MLPPredictor(ndim_out, edim, num_class, residual)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)
