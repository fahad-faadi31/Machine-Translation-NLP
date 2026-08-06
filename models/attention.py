import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()

        self.attention=nn.Linear(
            hidden_size*2,
            hidden_size
        )

        self.v=nn.Linear(
            hidden_size,
            1,
            bias=False
        )

    def forward(self,hidden,encoder_outputs):

        batch_size=encoder_outputs.shape[0]
        src_length=encoder_outputs.shape[1]

        hidden=hidden.unsqueeze(1).repeat(
            1,
            src_length,
            1
        )

        energy=torch.tanh(
            self.attention(
                torch.cat(
                    (
                        hidden,
                        encoder_outputs
                    ),
                    dim=2
                )
            )
        )

        attention=self.v(energy).squeeze(2)

        return torch.softmax(
            attention,
            dim=1
        )