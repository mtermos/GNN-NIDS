import torch
import torch.nn as nn
import torch.optim as optim


class CNN_LSTM(nn.Module):
    def __init__(
        self,
        # list[int]: channel sizes for each Conv1D layer, e.g. [16, 32]
        out_channels,
        seq_length,          # int: length of the input sequence
        num_classes,         # int: number of output classes
        cnn_use_bn=True,     # bool: whether to use BatchNorm in CNN
        lstm_hidden_size=64,  # int: hidden size of the LSTM
        lstm_num_layers=1,   # int: number of LSTM layers
        lstm_dropout=0.5,    # float: dropout in LSTM if lstm_num_layers > 1
        final_dropout=0.5,    # float: dropout before final classification layer
        model_name="cnn_lstm"
    ):
        # super(CNN_LSTM, self).__init__()
        super().__init__()
        self.model_name = model_name
        # ----------------------
        # 1) CNN feature extractor
        # ----------------------
        cnn_layers = []
        in_channels = 1  # input is (batch, 1, seq_length)

        for oc in out_channels:
            cnn_layers.append(
                nn.Conv1d(in_channels, oc, kernel_size=3, stride=1, padding=1))

            if cnn_use_bn:
                cnn_layers.append(nn.BatchNorm1d(oc))

            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(kernel_size=2))

            in_channels = oc

        self.cnn = nn.Sequential(*cnn_layers)

        # After each MaxPool1d(kernel_size=2), the sequence length is halved.
        # final_seq_len = seq_length / 2^(number_of_pools)
        self.final_seq_len = seq_length // (2 ** len(out_channels))

        # ----------------------
        # 2) LSTM
        # ----------------------
        # The LSTM input size = out_channels[-1]
        # We'll feed the CNN output (batch, out_channels[-1], final_seq_len)
        # into an LSTM whose input is (batch, final_seq_len, out_channels[-1])
        # => we will transpose before feeding into LSTM.
        self.lstm = nn.LSTM(
            input_size=out_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            # input/output shape: (batch, seq_len, input_size)
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )

        # ----------------------
        # 3) Classifier
        # ----------------------
        self.dropout = nn.Dropout(final_dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 1, seq_length)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # --- CNN ---
        # shape after CNN: (batch_size, out_channels[-1], final_seq_len)
        x = self.cnn(x)

        # Transpose to (batch_size, final_seq_len, out_channels[-1])
        x = x.transpose(1, 2)  # swap channel dim and sequence dim

        # --- LSTM ---
        # out shape: (batch_size, final_seq_len, lstm_hidden_size)
        # (hn, cn) have shape: (lstm_num_layers, batch_size, lstm_hidden_size)
        out, (hn, cn) = self.lstm(x)

        # We can take the last timestep's output from 'out'
        # shape: (batch_size, lstm_hidden_size)
        out = out[:, -1, :]

        # --- Classifier ---
        out = self.dropout(out)
        logits = self.fc(out)

        return logits


# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    # Hyperparameters
    # Two CNN layers: Conv1d(1->16), Conv1d(16->32)
    out_channels_list = [16, 32]
    seq_length = 64               # For example, if each sample is length 64
    num_classes = 2               # e.g., binary classification
    batch_size = 8

    model = CNN_LSTM(
        out_channels=out_channels_list,
        seq_length=seq_length,
        num_classes=num_classes,
        cnn_use_bn=True,
        lstm_hidden_size=64,
        lstm_num_layers=1,       # or 2, or more
        lstm_dropout=0.3,
        final_dropout=0.5
    )
    print(model)

    # Create a dummy input: (batch_size, 1, seq_length)
    sample_input = torch.randn(batch_size, 1, seq_length)
    output = model(sample_input)
    print("Output shape:", output.shape)  # (batch_size, num_classes)

    # Dummy training step
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    optimizer.zero_grad()
    preds = model(sample_input)
    labels = torch.randint(0, num_classes, (batch_size,))
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    print(f"Dummy CNN_LSTM training step complete. Loss: {loss.item():.4f}")
