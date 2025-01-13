# mdoel 4o, give it my code and told it to improve it

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class EGATLayer(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, num_heads, activation=F.relu, dropout=0.2):
        super(EGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Linear layers for node and edge features
        self.node_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.edge_fc = nn.Linear(edge_dim, out_dim * num_heads, bias=False)

        # Attention layers
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

        # Output projection
        self.output_fc = nn.Linear(out_dim * num_heads, out_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.node_fc.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.edge_fc.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.attn_fc.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.output_fc.weight,
                                gain=nn.init.calculate_gain('relu'))

    def edge_attention(self, edges):
        # Compute attention scores
        z_src = edges.src['z']
        z_dst = edges.dst['z']
        z_edge = edges.data['z']
        concat = th.cat([z_src + z_edge, z_dst + z_edge], dim=-1)
        e = self.attn_fc(concat)
        return {'e': F.leaky_relu(e)}

    def message_func(self, edges):
        # Attention scores and node-edge interactions
        return {'m': edges.data['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # Weighted sum of messages
        alpha = F.softmax(nodes.mailbox['e'], dim=1)  # Attention coefficients
        z = th.sum(alpha * nodes.mailbox['m'], dim=1)  # Weighted sum
        return {'z': z}

    def forward(self, graph, node_feats, edge_feats):
        with graph.local_scope():
            # Transform features
            node_feats_proj = self.node_fc(
                node_feats).view(-1, self.num_heads, self.out_dim)
            edge_feats_proj = self.edge_fc(
                edge_feats).view(-1, self.num_heads, self.out_dim)

            graph.ndata['z'] = node_feats_proj
            graph.edata['z'] = edge_feats_proj

            # Compute attention scores
            graph.apply_edges(self.edge_attention)

            # Message passing
            graph.update_all(self.message_func, self.reduce_func)

            # Update node features
            new_node_feats = graph.ndata['z'].view(
                -1, self.out_dim * self.num_heads)
            new_node_feats = self.output_fc(new_node_feats)

            if self.activation:
                new_node_feats = self.activation(new_node_feats)

            return self.dropout(new_node_feats)


class EGAT(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, num_heads, num_layers, activation=F.relu, dropout=0.2):
        super(EGAT, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            EGATLayer(in_dim, edge_dim, hidden_dim, num_heads, activation, dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                EGATLayer(hidden_dim, edge_dim, hidden_dim, num_heads, activation, dropout))

        # Output layer
        self.layers.append(
            EGATLayer(hidden_dim, edge_dim, out_dim, 1, None, dropout))

    def forward(self, graph, node_feats, edge_feats):
        for layer in self.layers:
            node_feats = layer(graph, node_feats, edge_feats)
        return node_feats


class EdgePredictor(nn.Module):
    def __init__(self, node_dim, edge_dim, output_dim, residual=False):
        super(EdgePredictor, self).__init__()
        self.residual = residual
        input_dim = node_dim * 2 + edge_dim if residual else node_dim * 2
        self.edge_fc = nn.Linear(input_dim, output_dim)

    def forward(self, graph, node_feats, edge_feats):
        with graph.local_scope():
            graph.ndata['h'] = node_feats
            graph.edata['h'] = edge_feats

            def apply_edges(edges):
                src, dst = edges.src['h'], edges.dst['h']
                if self.residual:
                    edge_feat = edges.data['h']
                    combined = th.cat([src, dst, edge_feat], dim=-1)
                else:
                    combined = th.cat([src, dst], dim=-1)
                return {'score': self.edge_fc(combined)}

            graph.apply_edges(apply_edges)
            return graph.edata['score']


class E_GAT(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, num_heads, num_layers, num_classes, activation=F.relu, dropout=0.2, residual=False):
        super(E_GAT, self).__init__()
        self.gnn = EGAT(in_dim, edge_dim, hidden_dim, out_dim,
                        num_heads, num_layers, activation, dropout)
        self.predictor = EdgePredictor(
            out_dim, edge_dim, num_classes, residual)

    def forward(self, graph, node_feats, edge_feats):
        node_feats = self.gnn(graph, node_feats, edge_feats)
        return self.predictor(graph, node_feats, edge_feats)
