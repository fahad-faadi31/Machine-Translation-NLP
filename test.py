import torch
import torch.nn as nn

from dataset import get_dataloaders
from vocabulary import Vocabulary
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq


device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


_,validation_loader,_,_=get_dataloaders(
    batch_size=16
)


checkpoint=torch.load(
    "best_model.pth",
    map_location=device
)


english_vocab=Vocabulary()
urdu_vocab=Vocabulary()

english_vocab.word2idx=checkpoint["english_word2idx"]
english_vocab.idx2word=checkpoint["english_idx2word"]

urdu_vocab.word2idx=checkpoint["urdu_word2idx"]
urdu_vocab.idx2word=checkpoint["urdu_idx2word"]


input_size=len(english_vocab)
output_size=len(urdu_vocab)

embedding_size=128
hidden_size=256
num_layers=2
dropout=0.3


encoder=Encoder(
    input_size,
    embedding_size,
    hidden_size,
    num_layers,
    dropout
)

decoder=Decoder(
    output_size,
    embedding_size,
    hidden_size,
    num_layers,
    dropout
)

model=Seq2Seq(
    encoder,
    decoder,
    device
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

criterion=nn.CrossEntropyLoss(
    ignore_index=0
)

model.eval()

validation_loss=0

with torch.no_grad():

    for source,target in validation_loader:

        source=source.to(device)
        target=target.to(device)

        output=model(
            source,
            target,
            teacher_force_ratio=0
        )

        output=output[:,1:].reshape(
            -1,
            output_size
        )

        target=target[:,1:].reshape(
            -1
        )

        loss=criterion(
            output,
            target
        )

        validation_loss+=loss.item()


average_validation_loss=validation_loss/len(validation_loader)

print(
    f"Validation Loss: {average_validation_loss:.4f}"
)