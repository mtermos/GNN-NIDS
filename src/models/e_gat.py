import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    # in_dim = 55
    # edim = 55
    # out_dim = z = 55 * 2

    def __init__(self, ndim_in, edim, ndim_out, activation):
        super(GATLayer, self).__init__()
        # equation (1)
#        print(edim)
        self.linear = nn.Linear(ndim_in + edim, ndim_out, bias=False)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        # equation (2)

        self.attn_fc = nn.Linear(2*ndim_in, 1)
        self.activation = activation
        self.reset_parameters()
        # print("OK1")

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        # print(edges.src["h"].shape)
        # print(edges.dst["h"].shape)
        # print(edges.data["h"].shape[2])
        # print("OKedge")
        z2 = th.cat([edges.src["h"], edges.dst["h"]], dim=2)
        # print("OKedge1")
        # print(z2.shape)
        a = self.attn_fc(z2)
        # print("OKedge2")
        # print(f'attention={F.leaky_relu(a)}')
        return {"e": F.leaky_relu(a)}

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        # print("OK2")
        # print("DONE8")
        # b=th.cat([edges.src['h'], edges.data['h']], 1)
        # print(b.shape)
        # print(edges.src['h'].shape)
        # print(edges.data['h'].shape)
        return {"m": self.linear(th.cat([edges.src['h'], edges.data['h']], 2)), "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        # print("DONE7")

        alpha = F.softmax(nodes.mailbox['e'], dim=1)
 #       print({nodes.mailbox['e']})
        # print(f'alpha={alpha.shape}')
       # print("DONE7")
        # equation (4)
#         z = th.mean(alpha * nodes.mailbox['m'],dim=1)
        z = th.sum(alpha * nodes.mailbox['m'], dim=1)
        # print(f'z.shape={z.shape}')
        # z = th.reshape(z, (z.shape[0], 1,z.shape[2]))
        # print("DONE9")
        # print

        return {'z': z}

        # print("DONE6")

    def forward(self, g, nfeats, efeats):
        with g.local_scope():
            g = g
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
          #  print("OK1")
            # equation (2)

            g.apply_edges(self.edge_attention)
         #   print("OK2")
            # equation (3) & (4)
            # self.g.update_all(self.message_func, self.reduce_func)
            g.update_all(self.message_func, self.reduce_func)

  #          print("OK3")
   #         print(g.ndata["z"])
    #        print(g.ndata["z"].shape)
     #       print(g.ndata["h"].shape)

            g.ndata['h'] = self.activation(self.W_apply(
                th.cat([g.ndata['h'], g.ndata['z']], 2)))
#            print(g.ndata['h'].shape)

            # g.ndata['h'] = th.reshape( g.ndata['h'], ( g.ndata['h'].shape[0], 1, g.ndata['h'].shape[2]))
            # g.ndata['h'] = g.ndata['h'][:, 0:1, :]
            # print(g.ndata['h'])
 #           print("OK4")
            return g.ndata['h']


class GAT(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, dropout):
        super().__init__()
       # self.layers = nn.ModuleList([
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(
                    GATLayer(ndim_in, edim, ndim_out[layer], activation))
            else:
                self.layers.append(
                    GATLayer(ndim_out[layer-1], edim, ndim_out[layer], activation))

        self.dropout = nn.Dropout(p=dropout)

       # print("DONE")

    def forward(self, g, nfeats, efeats):
        # for i, layer in enumerate(self.layers):
        #     nfeats = layer(g, nfeats, efeats)

        # print('hello')
        #         print(f"nfeats.shape: {nfeats.shape}")
        #         print(f"nfeats.sum(1).shape: {nfeats.sum(1).shape}")
        #         print("===================")
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
        self.gnn = GAT(ndim_in, edim, ndim_out, num_layers,
                       activation, dropout)

        self.pred = MLPPredictor(ndim_out[-1], edim, num_class, residual)
        # print("DONE")

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        # print("DONE")
        return self.pred(g, h)
