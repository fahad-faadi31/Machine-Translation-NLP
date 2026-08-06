from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from vocabulary import Vocabulary

class TranslationDataset(Dataset):
    def __init__(self,data,english_vocab,urdu_vocab,max_length=40):
        self.data=data
        self.english_vocab=english_vocab
        self.urdu_vocab=urdu_vocab
        self.max_length=max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        sample=self.data[index]

        english=sample["translation"]["en"].lower().strip()
        urdu=sample["translation"]["ur"].strip()

        source=[self.english_vocab.word2idx["<SOS>"]]
        source+=self.english_vocab.numericalize(english)
        source.append(self.english_vocab.word2idx["<EOS>"])

        target=[self.urdu_vocab.word2idx["<SOS>"]]
        target+=self.urdu_vocab.numericalize(urdu)
        target.append(self.urdu_vocab.word2idx["<EOS>"])

        return torch.tensor(source),torch.tensor(target)

def collate_fn(batch):
    source,target=zip(*batch)

    source=pad_sequence(
        source,
        batch_first=True,
        padding_value=0
    )

    target=pad_sequence(
        target,
        batch_first=True,
        padding_value=0
    )

    return source,target

def clean_dataset(dataset,max_length):
    cleaned=[]

    for sample in dataset:
        english=sample["translation"]["en"].lower().strip()
        urdu=sample["translation"]["ur"].strip()

        if english=="" or urdu=="":
            continue

        if len(english.split())>max_length:
            continue

        if len(urdu.split())>max_length:
            continue

        cleaned.append(sample)

    return cleaned

def get_dataloaders(
    batch_size=32,
    min_freq=2,
    max_samples=50000,
    max_length=40
):
    dataset=load_dataset(
        "Helsinki-NLP/opus-100",
        "en-ur"
    )

    train_data=dataset["train"]

    if max_samples<len(train_data):
        train_data=train_data.select(range(max_samples))

    train_data=clean_dataset(train_data,max_length)

    english_sentences=[]
    urdu_sentences=[]

    for sample in train_data:
        english_sentences.append(
            sample["translation"]["en"].lower().strip()
        )

        urdu_sentences.append(
            sample["translation"]["ur"].strip()
        )

    english_vocab=Vocabulary(min_freq)
    urdu_vocab=Vocabulary(min_freq)

    english_vocab.build_vocab(english_sentences)
    urdu_vocab.build_vocab(urdu_sentences)

    split=int(0.9*len(train_data))

    train_dataset=TranslationDataset(
        train_data[:split],
        english_vocab,
        urdu_vocab,
        max_length
    )

    validation_dataset=TranslationDataset(
        train_data[split:],
        english_vocab,
        urdu_vocab,
        max_length
    )

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    validation_loader=DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(validation_dataset)}")
    print(f"English Vocabulary: {len(english_vocab)}")
    print(f"Urdu Vocabulary: {len(urdu_vocab)}")

    return (
        train_loader,
        validation_loader,
        english_vocab,
        urdu_vocab
    )