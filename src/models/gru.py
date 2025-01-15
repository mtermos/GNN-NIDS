import torch
import torch.nn as nn
import torch.optim as optim


class NIDSGRU(nn.Module):
    def __init__(
        self,
        num_features,   # Input size (number of features in each sample)
        hidden_size,    # Hidden size of the GRU
        num_layers,     # Number of GRU layers
        num_classes,    # Output classes for classification
        use_bn=False,   # Whether to use Batch Normalization
        # Dropout probability (applied in GRU if num_layers > 1)
        dropout=0.5,
        model_name="gru"
    ):
        # super(NIDSGRU, self).__init__()
        super().__init__()

        self.model_name = model_name
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bn = use_bn

        # GRU
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        if self.use_bn:
            self.bn = nn.BatchNorm1d(hidden_size)

        self.fc = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len=1, num_features)
        Output shape: (batch_size, num_classes)
        """
        # GRU forward
        # out: (batch_size, seq_len, hidden_size)
        # hn:  (num_layers, batch_size, hidden_size)
        out, hn = self.gru(x)

        # Take the last time step's output
        # If seq_len=1, out[:, -1, :] => out[:, 0, :]
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # Apply BN if enabled
        if self.use_bn:
            out = self.bn(out)

        out = self.fc(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    # Example usage of NIDSGRU
    batch_size = 8
    num_features = 32
    hidden_size = 64
    num_layers = 2
    num_classes = 2

    model = NIDSGRU(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        use_bn=True,
        dropout=0.5
    )
    print(model)

    # Create dummy input: shape (batch_size, seq_len=1, num_features)
    sample_input = torch.randn(batch_size, 1, num_features)
    output = model(sample_input)
    print("GRU Output shape:", output.shape)  # (8, 2)

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
    print(f"Dummy GRU training step complete. Loss: {loss.item():.4f}")
