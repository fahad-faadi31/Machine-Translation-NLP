import torch
from vocabulary import Vocabulary
from models.seq2seq import Encoder,Decoder,Seq2Seq

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

input_size=len(english_vocab.word2idx)
output_size=len(urdu_vocab.word2idx)

embedding_size=64
hidden_size=128
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

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def translate(sentence,max_length=40):
    tokens=sentence.lower().strip().split()

    source=[english_vocab.word2idx["<SOS>"]]

    for token in tokens:
        source.append(
            english_vocab.word2idx.get(
                token,
                english_vocab.word2idx["<UNK>"]
            )
        )

    source.append(
        english_vocab.word2idx["<EOS>"]
    )

    source=torch.tensor(source).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden,cell=model.encoder(source)

    target=[urdu_vocab.word2idx["<SOS>"]]

    for _ in range(max_length):

        x=torch.tensor([target[-1]]).to(device)

        with torch.no_grad():
            output,hidden,cell=model.decoder(
                x,
                hidden,
                cell
            )

        best=output.argmax(1).item()

        if best==urdu_vocab.word2idx["<EOS>"]:
            break

        target.append(best)

    words=[]

    for index in target[1:]:
        word=urdu_vocab.idx2word.get(index,"<UNK>")

        if word not in ["<PAD>","<SOS>","<EOS>"]:
            words.append(word)

    return " ".join(words)

while True:

    sentence=input("Enter English sentence (or 'quit'): ")

    if sentence.lower()=="quit":
        break

    translation=translate(sentence)

    print(f"Urdu: {translation}\n")