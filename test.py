import torch
import torch.nn as nn
from dataset import get_dataloaders
from models.seq2seq import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, validation_loader, english_vocab, urdu_vocab = get_dataloaders()

input_size = len(english_vocab)
output_size = len(urdu_vocab)

embedding_size = 64
hidden_size = 128
num_layers = 2
dropout = 0.3

encoder = Encoder(
    input_size,
    embedding_size,
    hidden_size,
    num_layers,
    dropout
)

decoder = Decoder(
    output_size,
    embedding_size,
    hidden_size,
    num_layers,
    dropout
)

model = Seq2Seq(
    encoder,
    decoder,
    device
).to(device)

checkpoint = torch.load(
    "best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

criterion = nn.CrossEntropyLoss(ignore_index=0)

model.eval()

validation_loss = 0

with torch.no_grad():

    for source, target in validation_loader:

        source = source.to(device)
        target = target.to(device)

        output = model(
            source,
            target,
            teacher_force_ratio=0
        )

        output = output[:,1:].reshape(-1, output_size)
        target = target[:,1:].reshape(-1)

        loss = criterion(output, target)

        validation_loss += loss.item()

average_loss = validation_loss / len(validation_loader)

print(f"Validation Loss: {average_loss:.4f}")