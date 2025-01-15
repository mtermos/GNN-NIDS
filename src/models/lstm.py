import torch
import torch.nn as nn
import torch.optim as optim


class NIDSLSTM(nn.Module):
    def __init__(
        self,
        num_features,   # Input size (number of features in each sample)
        hidden_size,    # Hidden size of the LSTM
        num_layers,     # Number of LSTM layers
        num_classes,    # Output classes for classification
        use_bn=False,   # Whether to use Batch Normalization
        # Dropout probability (applied in LSTM if num_layers > 1)
        dropout=0.5,
        model_name="lstm"
    ):
        # super(NIDSLSTM, self).__init__()
        super().__init__()

        self.model_name = model_name
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bn = use_bn

        # LSTM: We treat each sample as a sequence of length=1, with 'num_features' as input_size
        # batch_first=True => input shape: (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Optional BatchNorm on the final hidden state
        if self.use_bn:
            self.bn = nn.BatchNorm1d(hidden_size)

        # Final fully-connected layer
        self.fc = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len=1, num_features)
        Output shape: (batch_size, num_classes)
        """
        # LSTM forward
        # out: (batch_size, seq_len, hidden_size)
        # (hn, cn): hidden and cell states with shape (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x)

        # We can take the output at the final timestep in the sequence.
        # If seq_len=1, out[:, -1, :] is simply out[:, 0, :] which is shape (batch_size, hidden_size)
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # If BN is used, apply on the final hidden vector
        if self.use_bn:
            out = self.bn(out)  # BN expects (batch_size, hidden_size)

        # Pass through fully-connected layer
        out = self.fc(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    # Example usage of NIDSLSTM
    batch_size = 8
    num_features = 32   # e.g. 32 features per sample
    hidden_size = 64
    num_layers = 2
    num_classes = 2     # e.g. binary classification

    model = NIDSLSTM(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        use_bn=True,    # Enable batch norm on the final hidden state
        dropout=0.5
    )
    print(model)

    # Create dummy input: shape (batch_size, seq_len=1, num_features)
    sample_input = torch.randn(batch_size, 1, num_features)
    output = model(sample_input)
    print("LSTM Output shape:", output.shape)  # (8, 2)

    # Example training step
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    optimizer.zero_grad()
    pred = model(sample_input)
    labels = torch.randint(0, num_classes, (batch_size,))
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()
    print(f"Dummy LSTM training step complete. Loss: {loss.item():.4f}")
