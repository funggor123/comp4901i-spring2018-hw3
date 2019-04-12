import pandas as pd
import re
import numpy as np
import pickle
import unicodedata
from torchtext import datasets
import torch
from torchtext.vocab import GloVe
import torch.utils.data as data
from torchtext import data
import gensim

PAD_INDEX = 0
UNK_INDEX = 1

contractions_dict = {
    "won't": "were not",
    "you'll": "you will",
    "we're": "we are",
    "that's": "that is",
    "were't": "were not",
    "i'd": "i do not",
    "i'll": "i will",
    "there's": "there is",
    "they'll": "they will",
    "it's": "it is",
    "they're": "they are",
    "i've": "i have",
    "we'll": "we will",
    "she's": "she is",
    "could": "could have",
    "we've": "we have",
    "you'd": "you don't",
    "you're": "you are",
    "they've": "they have",
    "shouldn't": "should not",
    "he's": "he is ",
    "should ve": "should have",
    "could've": "could have",
    "couldn't've": "could not have",
    "did n't": "did not",
    "do n't": "do not",
    "had n't": "had not",
    "had n't've": "had not have",
    "has n't": "has not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "should've": "should have",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "there'd": "here would",
    "there'd've": "there would have",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll've": "they will have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll've": "we will have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd've": "you would have",
    "you'll've": "you will have",
    "you've": "you have",
    "n't": "not",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "isn't": "is not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "i'm": "i am",
}


def clean(string):
    string = string.strip()
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, s)

    string = expand_contractions(string)
    string = re.sub(r'[^\w\s]', ' ', string)
    string = re.sub(r'[0-9\.]+', ' ', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]').sub('', string)
    return string

class Vocab():
    def __init__(self):
        self.word2Vector = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        self.no_of_vocab = len(self.word2Vector.wv.vectors)
        word2idx = {"PAD": 0}
        vocab_list = [(k, self.word2Vector.wv[k]) for k, v in self.word2Vector.wv.vocab.items()]
        self.embeddings_matrix = np.zeros((len(self.word2Vector.wv.vocab.items()) + 1, self.word2Vector.vector_size))
        for i in range(len(vocab_list)):
            word = vocab_list[i][0]
            word2idx[word] = i + 1
            self.embeddings_matrix[i + 1] = vocab_list[i][1]

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
        return [self.vocab.word2index[word] if word in self.vocab.word2index else
                self.vocab.word2index[self.vocab.word2index.most_similar(word)] for word in sentence]


def preprocess(filename, max_len=200, test=False):
    df = pd.read_csv(filename)
    id_ = []  # review id
    rating = []  # rating
    content = []  # review content

    for i in range(len(df)):
        id_.append(int(df['id'][i]))
        if not test:
            rating.append(int(df['rating'][i]))
        sentence = clean(str(df['content'][i]).strip())
        sentence = sentence.split()
        sent_len = len(sentence)
        # here we pad the sequence for whole training set, you can also try to do dynamic padding for each batch by customize collate_fn function
        # if you do dynamic padding and report it, we will give 1 points bonus
        if sent_len > max_len:
            content.append(sentence[:max_len])
        else:
            content.append(sentence + ["PAD"] * (max_len - sent_len))

    if test:
        len(id_) == len(content)
        return (id_, content, None)
    else:
        assert len(id_) == len(content) == len(rating)
        return (id_, content, rating)


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
    return data_loader_tr, data_loader_dev, data_loader_test, vocab.embeddings_matrix

