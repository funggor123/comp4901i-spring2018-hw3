
import torch
import torch.nn as nn
import torch.nn.functional as F



class WordCNN(nn.Module):

    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(WordCNN, self).__init__()
        #TO DO
        #hint useful function: nn.Embedding(), nn.Dropout(), nn.Linear(), nn.Conv1d() or nn.Conv2d(), 
        
    def forward(self, x):

        #TO DO
        #input x dim: (batch_size, max_seq_len, D)
        #output logit dim: (batch_size, num_classes)
        return logit
