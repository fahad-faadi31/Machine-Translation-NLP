from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from vocabulary import Vocabulary

class TranslationDataset(Dataset):
    def __init__(self,data,english_vocab,urdu_vocab):
        self.data=data
        self.english_vocab=english_vocab
        self.urdu_vocab=urdu_vocab

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

    source=pad_sequence(source,batch_first=True,padding_value=0)
    target=pad_sequence(target,batch_first=True,padding_value=0)

    return source,target

def get_dataloaders(batch_size=64,min_freq=2):
    dataset=load_dataset("Helsinki-NLP/opus-100","en-ur")

    train_data=dataset["train"]

    english_sentences=[]
    urdu_sentences=[]

    for sample in train_data:
        english_sentences.append(sample["translation"]["en"].lower().strip())
        urdu_sentences.append(sample["translation"]["ur"].strip())

    english_vocab=Vocabulary(min_freq)
    urdu_vocab=Vocabulary(min_freq)

    english_vocab.build_vocab(english_sentences)
    urdu_vocab.build_vocab(urdu_sentences)

    split=int(0.9*len(train_data))

    train_dataset=TranslationDataset(
        train_data.select(range(split)),
        english_vocab,
        urdu_vocab
    )

    test_dataset=TranslationDataset(
        train_data.select(range(split,len(train_data))),
        english_vocab,
        urdu_vocab
    )

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader,test_loader,english_vocab,urdu_vocab