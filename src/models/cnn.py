import torch
import torch.nn as nn
import torch.optim as optim


class NIDSCNN(nn.Module):
    def __init__(self, out_channels, num_features, num_classes, use_bn=True, dropout=0.5,
                 model_name="cnn"):
        """
        Args:
            out_channels (list): A list of integers specifying the out_channels for each Conv1D layer.
                                 e.g., [16, 32] -> two layers with 16 and 32 channels, respectively.
            num_features  (int): Number of input features per sample (length of the 1D input).
            num_classes   (int): Number of classes (for classification).
            use_bn       (bool): Whether to use Batch Normalization after each Conv1D.
            dropout     (float): Dropout probability in the classifier.
        """
        super().__init__()

        # super(NIDSCNN, self).__init__()
        self.model_name = model_name
        layers = []
        in_channels = 1  # Starting with 1 “channel” for the features

        # Build Convolutional Blocks Dynamically
        # for oc in out_channels:
        #     layers.append(nn.Conv1d(in_channels, oc,
        #                   kernel_size=3, stride=1, padding=1))

        #     # Apply Batch Normalization if specified
        #     if use_bn:
        #         layers.append(nn.BatchNorm1d(oc))

        #     layers.append(nn.ReLU())
        #     layers.append(nn.MaxPool1d(kernel_size=2))

        #     in_channels = oc

        # self.features = nn.Sequential(*layers)

        # # After each MaxPool1d(kernel_size=2), the feature length is halved.
        # # final_length = num_features // (2 ** (number_of_pooling_layers))
        # final_length = num_features // (2 ** len(out_channels))
        # in_feats = out_channels[-1] * final_length

        # # Build Classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_feats, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(64, num_classes)
        # )
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,  # Keras default is 'valid', which means no padding
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bn = nn.BatchNorm1d(64)
        # After the pool, we flatten. We'll do that in forward().
        # <-- you need to compute the in_features here
        in_features = 64 * ((num_features - 2) // 2)
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 1, input_dim)
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x


if __name__ == "__main__":
    # Example usage:
    out_channels_list = [16, 32, 64]
    num_features = 32   # e.g., 32 features in each input
    num_classes = 2     # e.g., binary classification

    # Create the model
    model = NIDSCNN(
        out_channels=out_channels_list,
        num_features=num_features,
        num_classes=num_classes,
        use_bn=True,        # Enable BatchNorm
        dropout=0.3         # Example dropout probability
    )
    print(model)

    # Example input (batch_size=8, channels=1, num_features=32)
    sample_input = torch.randn(8, 1, num_features)
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
