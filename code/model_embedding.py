import torch
import torch.nn as nn
import torch.nn.functional as F


class WordCNN(nn.Module):

    def __init__(self, args, embedding_matrix):
        super(WordCNN, self).__init__()
        # TO DO
        # hint useful function: nn.Embedding(), nn.Dropout(), nn.Linear(), nn.Conv1d() or nn.Conv2d(),

        kernel_sizes = args.kernel_sizes.split(",")
        self.ebd = int(args.embed_dim)
        self.num_filter = args.kernel_num

        embedding_matrix_tensor = torch.FloatTensor(embedding_matrix)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix_tensor)

        self.convW_1 = nn.Conv2d(1, self.num_filter, kernel_size=(int(kernel_sizes[0]), self.ebd))
        self.convW_2 = nn.Conv2d(1, self.num_filter, kernel_size=(int(kernel_sizes[1]), self.ebd))
        self.convW_3 = nn.Conv2d(1, self.num_filter, kernel_size=(int(kernel_sizes[2]), self.ebd))

        self.dropout = nn.Dropout(p=int(args.dropout))
        self.linear = nn.Linear(self.num_filter*3, int(args.class_num))


    def forward(self, x):
        # TO DO
        # input x dim: (batch_size, max_seq_len, D)
        # output logit dim: (batch_size, num_classes)

        ebd = self.embedding(x)
        ebd = ebd.unsqueeze(1)

        f1 = self.convW_1(ebd)
        f2 = self.convW_2(ebd)
        f3 = self.convW_3(ebd)

        output = F.max_pool2d(f1, kernel_size=(f1.shape[2], f1.shape[3]))
        output = F.relu(output)
        output = output.squeeze(3).squeeze(2)

        output2 = F.max_pool2d(f2, kernel_size=(f2.shape[2], f2.shape[3]))
        output2 = F.relu(output2)
        output2 = output2.squeeze(3).squeeze(2)

        output3 = F.max_pool2d(f3, kernel_size=(f3.shape[2], f3.shape[3]))
        output3 = F.relu(output3)
        output3 = output3.squeeze(3).squeeze(2)

        concat = torch.cat([output, output2, output3], 1)

        out = self.dropout(concat)
        out = self.linear(out)

        return out



