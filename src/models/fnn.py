import torch
import torch.nn as nn
import torch.optim as optim


class NIDSFNN(nn.Module):
    def __init__(self, hidden_units, num_features, num_classes, use_bn=True, dropout=0.5,
                 model_name="fnn"):
        """
        Args:
            hidden_units (list): A list of integers specifying the number of neurons in each hidden layer.
                                 e.g., [128, 64] -> two layers with 128 and 64 neurons, respectively.
            num_features  (int): Number of input features per sample.
            num_classes   (int): Number of classes (for classification).
            use_bn       (bool): Whether to use Batch Normalization after each hidden layer.
            dropout     (float): Dropout probability in the classifier.
        """
        super().__init__()

        self.model_name = model_name

        layers = []
        input_dim = num_features

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Final layer for classification
        layers.append(nn.Linear(input_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        return self.network(x)


if __name__ == "__main__":
    # Example usage:
    hidden_units = [128, 64]  # Hidden layers with 128 and 64 neurons
    num_features = 32         # e.g., 32 features in each input
    num_classes = 2           # e.g., binary classification

    # Create the model
    model = NIDSFNN(
        hidden_units=hidden_units,
        num_features=num_features,
        num_classes=num_classes,
        use_bn=True,        # Enable BatchNorm
        dropout=0.3         # Example dropout probability
    )
    print(model)

    # Example input (batch_size=8, num_features=32)
    sample_input = torch.randn(8, num_features)
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expect (8, num_classes)

    # Example training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy training loop (just one step for demonstration)
    model.train()
    optimizer.zero_grad()
    pred = model(sample_input)
    labels = torch.randint(0, num_classes, (8,))  # Random labels
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()
    print(f"Dummy training step completed. Loss: {loss.item():.4f}")
