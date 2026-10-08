import torch
import torch.nn as nn


class FraudSequenceGRU(nn.Module):
    """GRU over each cardholder's recent transaction history -> fraud probability."""

    def __init__(self, n_step_features: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_step_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_step_features)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        logit = self.head(last_hidden).squeeze(-1)
        return logit
