import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        device
    ):
        super().__init__()

        self.encoder=encoder
        self.decoder=decoder
        self.device=device


    def forward(
        self,
        source,
        target,
        teacher_force_ratio=0.5
    ):

        batch_size=source.shape[0]
        target_length=target.shape[1]

        target_vocab_size=self.decoder.output_size

        outputs=torch.zeros(
            batch_size,
            target_length,
            target_vocab_size
        ).to(self.device)


        encoder_outputs,hidden,cell=self.encoder(
            source
        )

        input=target[:,0]


        for t in range(1,target_length):

            output,hidden,cell=self.decoder(
                input,
                hidden,
                cell,
                encoder_outputs
            )

            outputs[:,t]=output

            best_guess=output.argmax(1)

            input=(
                target[:,t]
                if random.random()<teacher_force_ratio
                else best_guess
            )


        return outputs