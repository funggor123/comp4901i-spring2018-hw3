import torch
import torch.nn as nn
import torch.nn.functional as F


class WordCNN(nn.Module):

    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(WordCNN, self).__init__()
        # TO DO
        # hint useful function: nn.Embedding(), nn.Dropout(), nn.Linear(), nn.Conv1d() or nn.Conv2d(),

        kernel_sizes = args.kernel_sizes.split(",")
        self.embedding_dim = int(args.embed_dim)
        self.num_of_filters = args.kernel_num

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.conv1 = nn.Conv2d(1, self.num_of_filters, kernel_size=(int(kernel_sizes[0]), self.embedding_dim))
        self.conv2 = nn.Conv2d(1, self.num_of_filters, kernel_size=(int(kernel_sizes[1]), self.embedding_dim))
        self.conv3 = nn.Conv2d(1, self.num_of_filters, kernel_size=(int(kernel_sizes[2]), self.embedding_dim))

        self.dropout = nn.Dropout(p=int(args.dropout))
        self.linear = nn.Linear(self.num_of_filters * 3, int(args.class_num))

    def forward(self, x):
        # TO DO
        # input x dim: (batch_size, max_seq_len, D)
        # output logit dim: (batch_size, num_classes)

        ebd = self.embedding(x)
        ebd = ebd.unsqueeze(1)

        f1 = self.conv1(ebd)
        f2 = self.conv2(ebd)
        f3 = self.conv3(ebd)

        out1 = F.max_pool2d(f1, kernel_size=(f1.shape[2], f1.shape[3]))
        out1 = F.relu(out1)
        out1 = out1.squeeze(3).squeeze(2)

        out2 = F.max_pool2d(f2, kernel_size=(f2.shape[2], f2.shape[3]))
        out2 = F.relu(out2)
        out2 = out2.squeeze(3).squeeze(2)

        out3 = F.max_pool2d(f3, kernel_size=(f3.shape[2], f3.shape[3]))
        out3 = F.relu(out3)
        out3 = out3.squeeze(3).squeeze(2)

        concat = torch.cat([out1, out2, out3], 1)

        out = self.dropout(concat)
        out = self.linear(out)

        return out
