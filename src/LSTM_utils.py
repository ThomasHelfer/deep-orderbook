import torch.nn as nn
import torch.nn.functional as F


class DeepLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size=1, dropout_rate=0.5
    ):
        super(DeepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)

        self.dropout = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Use the output of the last time step
        last_time_step = lstm_out[:, -1, :]

        # Pass through the first linear layer, then apply ReLU and dropout
        out = self.dropout(F.relu(self.linear1(last_time_step)))

        # Finally, pass through the second linear layer to get the prediction
        y_pred = self.linear2(out)

        return y_pred
