import pandas as pd
import re
import numpy as np
import pickle

import torch
import torch.utils.data as data

PAD_INDEX = 0
UNK_INDEX = 1
def clean(sent):
    # clean the data
    ############################################################
    # TO DO
    ############################################################
    return sent

class Vocab():
    def __init__(self):
        self.word2index = {"PAD":PAD_INDEX ,"UNK":UNK_INDEX }
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK" } 
        self.n_words = 2 # Count default tokens
        self.word_num = 0
    def index_words(self, sentence):
        for word in sentence:
            self.word_num+=1
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words+=1
            else:
                self.word2count[word]+=1

def Lang(vocab, file_name):
    statistic = {"sent_num":0, "word_num":0, "vocab_size":0, "max_len":0, "avg_len":0, "len_std":0, "class_distribution":{} }
    df = pd.read_csv(file_name)
    statistic["sent_num"] = len(df)
    sent_len_list = []
    ############################################################
    # TO DO
    #build vocabulary and statistic
    

    ############################################################
    return vocab, statistic


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        self.id, self.X, self.y = data
        self.vocab = vocab
        self.num_total_seqs = len(self.X)
        self.id = torch.LongTensor(self.id)
        if(self.y is not None):self.y = torch.LongTensor(self.y)
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ind = self.id[index]
        X = self.tokenize(self.X[index])
        if(self.y is not None):
            y = self.y[index]
            return torch.LongTensor(X), y, ind
        else:
            return torch.LongTensor(X), ind

    def __len__(self):
        return self.num_total_seqs
    def tokenize(self, sentence):
        return [self.vocab.word2index[word] if word in self.vocab.word2index else UNK_INDEX for word in sentence]

def preprocess(filename, max_len=200, test=False):
    df = pd.read_csv(filename)
    id_ = [] # review id
    rating = [] # rating
    content = [] #review content

    for i in range(len(df)):
        id_.append(int(df['id'][i]))
        if not test:
            rating.append(int(df['rating'][i]))
        sentence = clean(str(df['content'][i]).strip())
        sentence = sentence.split()
        sent_len = len(sentence)
        # here we pad the sequence for whole training set, you can also try to do dynamic padding for each batch by customize collate_fn function
        # if you do dynamic padding and report it, we will give 1 points bonus
        if sent_len>max_len:
            content.append(sentence[:max_len])
        else:
            content.append(sentence+["PAD"]*(max_len-sent_len))
        
    if test:
        len(id_) == len(content)
        return (id_, content, None)
    else:
        assert len(id_) == len(content) ==len(rating)
        return (id_, content, rating)

def get_dataloaders(batch_size, max_len):
    vocab = Vocab()
    vocab, statistic = Lang(vocab, "train.csv")

    train_data = preprocess("train.csv", max_len)
    dev_data = preprocess("dev.csv", max_len)
    test_data = preprocess("test.csv",max_len, test=True)
    train = Dataset(train_data, vocab)
    dev = Dataset(dev_data, vocab)
    test = Dataset(test_data, vocab)
    print(statistic)
    data_loader_tr = torch.utils.data.DataLoader(dataset=train,
                                                    batch_size=batch_size, 
                                                    shuffle=True)
    data_loader_dev = torch.utils.data.DataLoader(dataset=dev,
                                                    batch_size=batch_size, 
                                                    shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=test,
                                                    batch_size=batch_size, 
                                                    shuffle=False)
    return data_loader_tr, data_loader_dev, data_loader_test, statistic["vocab_size"]