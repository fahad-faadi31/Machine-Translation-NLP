import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models.seq2seq import Encoder,Decoder,Seq2Seq

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader,validation_loader,english_vocab,urdu_vocab=get_dataloaders()

input_size=len(english_vocab)
output_size=len(urdu_vocab)

embedding_size=64
hidden_size=128
num_layers=2
dropout=0.3
learning_rate=0.001
epochs=10

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

optimizer=optim.Adam(model.parameters(),lr=learning_rate)

criterion=nn.CrossEntropyLoss(ignore_index=0)

best_validation_loss=float("inf")

for epoch in range(epochs):

    model.train()

    train_loss=0

    for source,target in train_loader:

        source=source.to(device)
        target=target.to(device)

        optimizer.zero_grad()

        output=model(source,target)

        output=output[:,1:].reshape(-1,output_size)
        target=target[:,1:].reshape(-1)

        loss=criterion(output,target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1
        )

        optimizer.step()

        train_loss+=loss.item()

    average_train_loss=train_loss/len(train_loader)

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

            output=output[:,1:].reshape(-1,output_size)
            target=target[:,1:].reshape(-1)

            loss=criterion(output,target)

            validation_loss+=loss.item()

    average_validation_loss=validation_loss/len(validation_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {average_train_loss:.4f} "
        f"Validation Loss: {average_validation_loss:.4f}"
    )

    if average_validation_loss<best_validation_loss:

        best_validation_loss=average_validation_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "english_word2idx": english_vocab.word2idx,
                "english_idx2word": english_vocab.idx2word,
                "urdu_word2idx": urdu_vocab.word2idx,
                "urdu_idx2word": urdu_vocab.idx2word
            },
            "best_model.pth"
        )

        print("Best model saved!")