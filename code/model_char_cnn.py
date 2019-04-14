import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):

    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(CharCNN, self).__init__()
        # TO DO
        # hint useful function: nn.Embedding(), nn.Dropout(), nn.Linear(), nn.Conv1d() or nn.Conv2d(),

        self.embedding_dim = int(args.embed_dim)

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(7, self.embedding_dim), stride=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(5120, 1024),
            nn.ReLU(),
            nn.Dropout(p=int(args.dropout))
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=int(args.dropout))
        )

        self.fc3 = nn.Linear(1024, int(args.class_num))
        self.dropout = nn.Dropout(p=int(args.dropout))

    def forward(self, x):
        # TO DO
        # input x dim: (batch_size, max_seq_len, D)
        # output logit dim: (batch_size, num_classes)

        ebd = self.embedding(x)
        ebd = ebd.unsqueeze(1)
        out = self.conv1(ebd)
        out = out.squeeze(3)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        # collapse
        x = out.view(out.size(0), -1)

        # linear layer
        x = self.fc1(x)
        x = self.dropout(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)

        return x
