import random
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,dropout):
        super().__init__()

        self.embedding=nn.Embedding(input_size,embedding_size)
        self.lstm=nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        embedding=self.dropout(self.embedding(x))
        outputs,(hidden,cell)=self.lstm(embedding)
        return hidden,cell

class Decoder(nn.Module):
    def __init__(self,output_size,embedding_size,hidden_size,num_layers,dropout):
        super().__init__()

        self.embedding=nn.Embedding(output_size,embedding_size)
        self.lstm=nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc=nn.Linear(hidden_size,output_size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,hidden,cell):
        x=x.unsqueeze(1)

        embedding=self.dropout(self.embedding(x))

        outputs,(hidden,cell)=self.lstm(embedding,(hidden,cell))

        predictions=self.fc(outputs.squeeze(1))

        return predictions,hidden,cell

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device):
        super().__init__()

        self.encoder=encoder
        self.decoder=decoder
        self.device=device

    def forward(self,source,target,teacher_force_ratio=0.5):
        batch_size=source.shape[0]
        target_length=target.shape[1]
        target_vocab_size=self.decoder.fc.out_features

        outputs=torch.zeros(
            batch_size,
            target_length,
            target_vocab_size
        ).to(self.device)

        hidden,cell=self.encoder(source)

        x=target[:,0]

        for t in range(1,target_length):
            output,hidden,cell=self.decoder(x,hidden,cell)

            outputs[:,t]=output

            best_guess=output.argmax(1)

            if random.random()<teacher_force_ratio:
                x=target[:,t]
            else:
                x=best_guess

        return outputs