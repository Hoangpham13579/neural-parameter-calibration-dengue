import torch
from torch import nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    OPTIMIZERS = {
        "Adagrad": torch.optim.Adagrad,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SparseAdam": torch.optim.SparseAdam,
        "Adamax": torch.optim.Adamax,
        "ASGD": torch.optim.ASGD,
        "LBFGS": torch.optim.LBFGS,
        "NAdam": torch.optim.NAdam,
        "RAdam": torch.optim.RAdam,
        "RMSprop": torch.optim.RMSprop,
        "Rprop": torch.optim.Rprop,
        "SGD": torch.optim.SGD,
    }

    def __init__(
        self,
        conv_input: int,
        lstm_input_size: int,
        hidden_size: int,
        num_lstm_layers: int,
        attention_heads: int,
        output_size: int,
        batch_size: int,
        optimizer: str = "Adam",
        learning_rate: float = 0.002,
        optimizer_kwargs: dict = {},
        use_cnnlstm: bool = True,
    ):
        super(CNN_LSTM, self).__init__()

        # Define variables
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # 1. Enhanced CNN feature extraction
        self.conv1 = nn.Conv1d(conv_input, 32, kernel_size=3, dilation=2, padding=2)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        # 2. Bidirectional LSTM - outputs with 2xhidden_size
        self.LSTM = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # 3. Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=attention_heads,
            batch_first=True,
        )

        # 4. Layer normalization (helps with attention)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # 5. Fully connected layer - adjusted for bidirectional
        self.fc = nn.Linear(2 * hidden_size, output_size)

        # Define the optimizer and learning rate
        self.optimizer = self.OPTIMIZERS[optimizer](
            self.parameters(), lr=learning_rate, **optimizer_kwargs
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        # Capture local patterns with CNN layers
        x = self.conv1(x)
        x = F.selu(x)
        x = self.conv2(x)
        # x = self.batch_norm1(x)
        x = F.selu(x)  # (16,1)
        x = self.pool1(x)

        # Bidirectional LSTM captures past and future depencencies
        # Shape designed based on official PyTorch LSTM (link: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
        out, _ = self.LSTM(x)

        # # Self-attention mechanism _ each attention head focuses on each latent variable & capture dependencies bewteen latent variables
        # attn_output, _ = self.attention(
        #     lstm_out, lstm_out, lstm_out
        # )

        # # Residual connection and normalization _ improves gradient flow during training, which is important for stability when dealing with ODE parameter
        # out = self.layer_norm(lstm_out + attn_output)

        # Fully connected layer
        if len(out.shape) == 3:
            out = out[:, -1, :]
            # flatten before passing to FC layer
            out = torch.reshape(
                out, (out.shape[0] * out.shape[1],)
            )  # (2*batch_size*hidden_size,)
        else:
            out = torch.sum(out, axis=0)  # (2*hidden_size,)

        out = torch.abs(self.fc(out))
        return out
