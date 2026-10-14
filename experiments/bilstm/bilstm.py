"""Bi-LSTM written from scratch (no nn.LSTM) for the drought teacher comparison.

Each direction has its own W (input -> hidden), U (hidden -> hidden), and b for the
forget, input, limbo (candidate), and output gates.
"""

import lightning as L
import torch
from torch import nn

GATES = ("forget", "input", "limbo", "output")
DIRECTIONS = ("fwd", "bwd")


class BiLSTM(L.LightningModule):
    def __init__(self, input_size, hidden_size, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Same init as nn.LSTM: U(-1/sqrt(H), 1/sqrt(H)).
        bound = hidden_size**-0.5

        def param(*shape):
            return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))

        self.W = nn.ParameterDict()
        self.U = nn.ParameterDict()
        self.b = nn.ParameterDict()
        for direction in DIRECTIONS:
            for gate in GATES:
                key = f"{direction}_{gate}"
                self.W[key] = param(input_size, hidden_size)
                self.U[key] = param(hidden_size, hidden_size)
                self.b[key] = param(hidden_size)

        self.head = nn.Linear(2 * hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def _gate(self, gate, input_value, short_memory, direction):
        key = f"{direction}_{gate}"
        return input_value @ self.W[key] + short_memory @ self.U[key] + self.b[key]

    def lstm_unit(self, input_value, long_memory, short_memory, direction):
        f = torch.sigmoid(self._gate("forget", input_value, short_memory, direction))
        i = torch.sigmoid(self._gate("input", input_value, short_memory, direction))
        l = torch.tanh(self._gate("limbo", input_value, short_memory, direction))  # noqa: E741
        o = torch.sigmoid(self._gate("output", input_value, short_memory, direction))

        long_memory = f * long_memory + i * l
        short_memory = o * torch.tanh(long_memory)
        return long_memory, short_memory

    def forward(self, input):  # (batch, seq_len, input_size)
        batch, seq_len, _ = input.shape
        zeros = input.new_zeros(batch, self.hparams.hidden_size)

        long_f, short_f = zeros, zeros
        for t in range(seq_len):
            long_f, short_f = self.lstm_unit(input[:, t], long_f, short_f, "fwd")

        long_b, short_b = zeros, zeros
        for t in reversed(range(seq_len)):
            long_b, short_b = self.lstm_unit(input[:, t], long_b, short_b, "bwd")

        return self.head(torch.cat([short_f, short_b], dim=1))  # (batch, 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", self.loss_fn(self(x), y), on_step=False, on_epoch=True)

    def to_torch_lstm(self):
        """Copy these weights into an equivalent nn.LSTM (gate order i, f, g, o)."""
        H, D = self.hparams.hidden_size, self.hparams.input_size
        ref = nn.LSTM(D, H, batch_first=True, bidirectional=True).to(self.dtype)
        order = ("input", "forget", "limbo", "output")
        with torch.no_grad():
            for direction, suffix in (("fwd", "l0"), ("bwd", "l0_reverse")):
                keys = [f"{direction}_{g}" for g in order]
                # Ours computes x @ W; nn.LSTM computes W_ih @ x, hence the transpose.
                getattr(ref, f"weight_ih_{suffix}").copy_(torch.cat([self.W[k].T for k in keys]))
                getattr(ref, f"weight_hh_{suffix}").copy_(torch.cat([self.U[k].T for k in keys]))
                # nn.LSTM adds bias_ih + bias_hh; we hold the sum in one bias.
                getattr(ref, f"bias_ih_{suffix}").copy_(torch.cat([self.b[k] for k in keys]))
                getattr(ref, f"bias_hh_{suffix}").zero_()
        return ref
