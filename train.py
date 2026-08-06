import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models.seq2seq import Encoder,Decoder,Seq2Seq

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader,test_loader,english_vocab,urdu_vocab=get_dataloaders()

input_size=len(english_vocab)
output_size=len(urdu_vocab)

embedding_size=256
hidden_size=512
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

best_loss=float("inf")

for epoch in range(epochs):
    model.train()

    total_loss=0

    for source,target in train_loader:
        source=source.to(device)
        target=target.to(device)

        optimizer.zero_grad()

        output=model(source,target)

        output=output[:,1:].reshape(-1,output_size)
        target=target[:,1:].reshape(-1)

        loss=criterion(output,target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1)

        optimizer.step()

        total_loss+=loss.item()

    average_loss=total_loss/len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {average_loss:.4f}")

    if average_loss<best_loss:
        best_loss=average_loss

        torch.save(
            {
                "model_state_dict":model.state_dict(),
                "english_vocab":english_vocab,
                "urdu_vocab":urdu_vocab
            },
            "best_model.pth"
        )

        print("Best model saved!")