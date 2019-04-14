import pandas as pd
import unicodedata
import numpy as np
import torch
import torch.utils.data as data

PAD_INDEX = 0
UNK_INDEX = 1

'''
Vocab Class for Char-CNN
'''


class Vocab():
    def __init__(self):
        alphabet = " abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        self.char_dict = {"UNK": UNK_INDEX}
        self.default_words = 1
        self.no_vocab = 1
        for i, char in enumerate(alphabet):
            self.char_dict[char] = self.default_words + i
            self.no_vocab += 1


def clean(string):
    string = string.strip()
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return string


def preprocess(filename, max_len=4000, test=False):
    df = pd.read_csv(filename)
    id_ = []  # review id
    rating = []  # rating
    content = []  # review content
    sent_len_list = []

    for i in range(len(df)):
        id_.append(int(df['id'][i]))
        if not test:
            rating.append(int(df['rating'][i]))
        sentence = clean(str(df['content'][i]))
        sentence = list(sentence)
        sent_len = len(sentence)
        sent_len_list.append(sent_len)

        if sent_len > max_len:
            content.append(sentence[:max_len])
        else:
            content.append(sentence + [" "] * (max_len - sent_len))

    # 6. Sentence Length Mean
    print("avg " + str(np.array(sent_len_list).mean()))

    # 7. Sentence Length Std
    print("std " + str(np.array(sent_len_list).std()))
    if test:
        len(id_) == len(content)
        return (id_, content, None)
    else:
        assert len(id_) == len(content) == len(rating)
        return (id_, content, rating)


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        self.id, self.X, self.y = data
        self.vocab = vocab
        self.num_total_seqs = len(self.X)
        self.id = torch.LongTensor(self.id)
        if (self.y is not None): self.y = torch.LongTensor(self.y)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ind = self.id[index]
        X = self.tokenize(self.X[index])
        if (self.y is not None):
            y = self.y[index]
            return torch.LongTensor(X), y, ind
        else:
            return torch.LongTensor(X), ind

    def __len__(self):
        return self.num_total_seqs

    def tokenize(self, sentence):
        return [self.vocab.char_dict[char] if char in self.vocab.char_dict else UNK_INDEX for char in sentence]


def get_dataloaders(batch_size, max_len):
    vocab = Vocab()

    train_data = preprocess("train.csv", max_len)
    dev_data = preprocess("dev.csv", max_len)
    test_data = preprocess("test.csv", max_len, test=True)
    train = Dataset(train_data, vocab)
    dev = Dataset(dev_data, vocab)
    test = Dataset(test_data, vocab)

    data_loader_tr = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=batch_size,
                                                 shuffle=True)
    data_loader_dev = torch.utils.data.DataLoader(dataset=dev,
                                                  batch_size=batch_size,
                                                  shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=test,
                                                   batch_size=batch_size,
                                                   shuffle=False)
    return data_loader_tr, data_loader_dev, data_loader_test, vocab.no_vocab
