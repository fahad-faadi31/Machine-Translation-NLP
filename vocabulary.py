from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=2, max_size=8000):
        self.min_freq = min_freq
        self.max_size = max_size

        self.word2idx = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }

        self.idx2word = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

    def build_vocab(self, sentences):
        counter = Counter()

        for sentence in sentences:
            counter.update(sentence.strip().split())

        words = [
            (word, freq)
            for word, freq in counter.items()
            if freq >= self.min_freq
        ]

        words = sorted(
            words,
            key=lambda x: x[1],
            reverse=True
        )

        words = words[: self.max_size]

        index = len(self.word2idx)

        for word, _ in words:
            self.word2idx[word] = index
            self.idx2word[index] = word
            index += 1

    def numericalize(self, sentence):
        return [
            self.word2idx.get(word, self.word2idx["<UNK>"])
            for word in sentence.strip().split()
        ]

    def decode(self, indices):
        words = []

        for index in indices:
            word = self.idx2word.get(index, "<UNK>")

            if word == "<EOS>":
                break

            if word not in ["<PAD>", "<SOS>"]:
                words.append(word)

        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)