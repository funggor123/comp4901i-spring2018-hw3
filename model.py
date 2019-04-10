
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WordCNN(nn.Module):

    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(WordCNN, self).__init__()
        #TO DO
        #hint useful function: nn.Embedding(), nn.Dropout(), nn.Linear(), nn.Conv1d() or nn.Conv2d(), 
        print("This is args", args)
        #self.word_embeddings = nn.Embedding(vocab_size, int(args.embed_dim))
        #for word embedding
        self.word_embeddings = nn.Embedding(vocab_size, int(args.embed_dim))
        self.word_embeddings.weight.data.copy_(embedding_matrix)
        #print(embedding_matrix.shape)
        #self.word_embeddings.weight.data.copy_(torch.from_numpy(np.ndarray(embedding_matrix)))
        
        mytuple = args.kernel_sizes.split(",")
        print("kernel size", mytuple)
        # 1 is input layer, 100 is the / kernel size = kernal size*100
        kernel_sizes_array = args.kernel_sizes.split(',')
        tup1 = (int(kernel_sizes_array[0]),int(args.embed_dim)) 
        tup2 = (int(kernel_sizes_array[1]),int(args.embed_dim))
        tup3 = (int(kernel_sizes_array[2]),int(args.embed_dim)) 

        self.conv1 = nn.Conv2d(1, args.kernel_num,tup1)
        self.conv2 = nn.Conv2d(1, args.kernel_num,tup2)
        self.conv3 = nn.Conv2d(1, args.kernel_num,tup3)
        self.dropout = nn.Dropout(p=int(args.dropout))
        # Linear 1 
        self.linear = nn.Linear(int(args.kernel_num)*3, int(args.class_num))
        
    def forward(self, x):

        #TO DO
        #input x dim: (batch_size, max_seq_len, D)
        #output logit dim: (batch_size, num_classes)
        
        '''
        word_embeddings = self.word_embeddings(x).unsqueeze(1)
        conved_0 = F.relu(self.conv1(word_embeddings))
        conved_1 = F.relu(self.conv2(word_embeddings))
        conved_2 = F.relu(self.conv3(word_embeddings))

        pooled_0 = F.max_pool2d(conved_0, kernel_size=(conved_0.shape[2],conved_0.shape[3])).squeeze(3).squeeze(2)
        pooled_1 = F.max_pool2d(conved_1, kernel_size=(conved_1.shape[2],conved_1.shape[3])).squeeze(3).squeeze(2)
        pooled_2 = F.max_pool2d(conved_2, kernel_size=(conved_2.shape[2],conved_2.shape[3])).squeeze(3).squeeze(2)

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
        
        logit = self.linear(cat) 
        '''
        
        # pre trained embedding
        #text = x.permute(1, 0)
        word_embeddings = self.word_embeddings(x).unsqueeze(1)
        conved_0 = F.relu(self.conv1(word_embeddings))
        conved_1 = F.relu(self.conv2(word_embeddings))
        conved_2 = F.relu(self.conv3(word_embeddings))

       # pooled_0 = F.max_pool2d(conved_0, conved_0.shape[2]).squeeze(3).squeeze(2)
        #pooled_1 = F.max_pool2d(conved_1, conved_1.shape[2]).squeeze(3).squeeze(2)
        #pooled_2 = F.max_pool2d(conved_2, conved_2.shape[2]).squeeze(3).squeeze(2)
        pooled_0 = F.max_pool2d(conved_0, kernel_size=(conved_0.shape[2],conved_0.shape[3])).squeeze(3).squeeze(2)
        pooled_1 = F.max_pool2d(conved_1, kernel_size=(conved_1.shape[2],conved_1.shape[3])).squeeze(3).squeeze(2)
        pooled_2 = F.max_pool2d(conved_2, kernel_size=(conved_2.shape[2],conved_2.shape[3])).squeeze(3).squeeze(2)

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
        
        logit = self.linear(cat)
        
        return logit