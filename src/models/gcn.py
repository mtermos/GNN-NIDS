import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv


# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MLP for prediction on the output of readout.
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn

# pylint: disable=W0221


class MLPPredictor(nn.Module):
    """Two-layer MLP for regression or soft classification
    over multiple tasks from graph representations.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input graph features
    hidden_feats : int
        Number of graph features in hidden layers
    n_classes : int
        Number of classes, which is also the output size.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """

    def __init__(self, in_feats, hidden_feats, n_classes, dropout=0.):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_classes),
            nn.Sigmoid()
        )

    def forward(self, feats):
        """Make prediction.

        Parameters
        ----------
        feats : FloatTensor of shape (B, M3)
            * B is the number of graphs in a batch
            * M3 is the input graph feature size, must match in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
        """
        return self.predict(feats)

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
