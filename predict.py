import torch

from vocabulary import Vocabulary
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

model.eval()

def translate(sentence,max_length=40):

    tokens=[
        english_vocab.word2idx["<SOS>"]
    ]

    tokens+=english_vocab.numericalize(
        sentence.lower()
    )

    tokens.append(
        english_vocab.word2idx["<EOS>"]
    )

    source=torch.tensor(
        tokens
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        encoder_outputs,hidden,cell=model.encoder(
            source
        )

    input_token=torch.tensor(
        [urdu_vocab.word2idx["<SOS>"]]
    ).to(device)

    translated=[]

    for _ in range(max_length):

        with torch.no_grad():

            output,hidden,cell=model.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs
            )

        prediction=output.argmax(1).item()

        if prediction==urdu_vocab.word2idx["<EOS>"]:
            break

        translated.append(
            urdu_vocab.idx2word.get(
                prediction,
                "<UNK>"
            )
        )

        input_token=torch.tensor(
            [prediction]
        ).to(device)

    return " ".join(translated)


while True:

    sentence=input(
        "Enter English sentence: "
    )

    if sentence.lower()=="exit":
        break

    translation=translate(sentence)

    print(
        "Urdu:",
        translation
    )