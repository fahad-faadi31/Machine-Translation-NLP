import torch
import torch.nn as nn
from models.attention import Attention


class Decoder(nn.Module):

    def __init__(
        self,
        output_size,
        embedding_size,
        hidden_size,
        num_layers,
        dropout
    ):
        super().__init__()

        self.output_size=output_size
        self.hidden_size=hidden_size

        self.embedding=nn.Embedding(
            output_size,
            embedding_size
        )

        self.attention=Attention(
            hidden_size
        )

        self.rnn=nn.LSTM(
            embedding_size+hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc_out=nn.Linear(
            hidden_size*2+embedding_size,
            output_size
        )

        self.dropout=nn.Dropout(dropout)


    def forward(
        self,
        input,
        hidden,
        cell,
        encoder_outputs
    ):

        input=input.unsqueeze(1)

        embedded=self.dropout(
            self.embedding(input)
        )

        attention_weights=self.attention(
            hidden[-1],
            encoder_outputs
        )

        attention_weights=attention_weights.unsqueeze(1)

        context=torch.bmm(
            attention_weights,
            encoder_outputs
        )

        rnn_input=torch.cat(
            (
                embedded,
                context
            ),
            dim=2
        )

        output,(hidden,cell)=self.rnn(
            rnn_input,
            (hidden,cell)
        )

        output=output.squeeze(1)

        context=context.squeeze(1)

        prediction=self.fc_out(
            torch.cat(
                (
                    output,
                    context,
                    embedded.squeeze(1)
                ),
                dim=1
            )
        )

        return prediction,hidden,cell