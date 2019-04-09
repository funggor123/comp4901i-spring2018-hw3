
import torch
import torch.nn as nn
import torch.nn.functional as F


class WordCNN(nn.Module):

    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(WordCNN, self).__init__()
        #TO DO
        kernel_size = args.kernel_sizes.split(',')
        self.embed = nn.Embedding(vocab_size, args.embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(args.embed_dim, args.kernel_num, kernel_size=int(k)) for k in kernel_size])

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(args.embed_dim, args.kernel_num, kernel_size=kernel_size[0],stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=kernel_size[0])
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(args.embed_dim, args.kernel_num, kernel_size=kernel_size[1], stride=1),
        #     nn.ReLU()
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(args.embed_dim, args.kernel_num, kernel_size=kernel_size[2], stride=1),
        #     nn.ReLU()
        # )
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(kernel_size)*args.kernel_num, args.class_num)
        #hint useful function: nn.Embedding(), nn.Dropout(), nn.Linear(), nn.Conv1d() or nn.Conv2d(),

    def forward(self, x):
        #TO DO
        #input x dim: (batch_size, max_seq_len, D)
        #output logit dim: (batch_size, num_classes)
        embed = self.embed(x)
        embed = embed.permute(0, 2, 1)
        logit = [F.relu(conv(embed)) for conv in self.convs]
        logit = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in logit]
        logit = self.dropout(torch.cat(logit, dim=1))
        return F.softmax(self.linear(logit))
