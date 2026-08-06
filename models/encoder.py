import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        num_layers,
        dropout
    ):
        super().__init__()

        self.embedding=nn.Embedding(
            input_size,
            embedding_size
        )

        self.rnn=nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout=nn.Dropout(dropout)


    def forward(self,source):

        embedded=self.dropout(
            self.embedding(source)
        )

        outputs,(hidden,cell)=self.rnn(
            embedded
        )

        return outputs,hidden,cell