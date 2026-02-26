import torch
from torch.utils.data import Dataset
from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

    def build(self, tokenized_texts):
        counter = Counter()

        for tokens in tokenized_texts:
            counter.update(tokens)

        words = [w for w, f in counter.items() if f >= self.min_freq]

        # 0 = PAD, 1 = UNK
        self.word2idx = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }

        for w in words:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.word2idx)

        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, tokens):
        return [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in tokens]

    def __len__(self):
        return len(self.word2idx)


class RNNDataset(Dataset):
    def __init__(self, tokenized_texts, labels, vocab, max_len=200):
        self.vocab = vocab
        self.max_len = max_len
        self.labels = labels

        self.labels = labels.values if hasattr(labels, "values") else labels

        self.encoded = [
            self.pad(self.vocab.encode(tokens)) for tokens in tokenized_texts
        ]

    def pad(self, seq):
        if len(seq) > self.max_len:
            return seq[: self.max_len]
        return seq + [0] * (self.max_len - len(seq))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )

class SequenceEncoder:

    def __init__(self, vocab, tokenizer, max_len=200):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode_text(self, text):
        tokens = self.tokenizer(text)

        encoded = [
            self.vocab.word2idx.get(w, self.vocab.word2idx["<UNK>"])
            for w in tokens
        ]

        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        else:
            encoded += [0] * (self.max_len - len(encoded))

        return encoded

    def batch_encode(self, texts):
        encoded = [self.encode_text(t) for t in texts]
        return torch.tensor(encoded, dtype=torch.long)