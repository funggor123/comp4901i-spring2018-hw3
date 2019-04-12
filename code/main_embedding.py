import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import classification_report, accuracy_score

from preprocess_embedding import get_dataloaders
from model_embedding import WordCNN

use_gpu = False

def trainer(train_loader,dev_loader, model, optimizer, criterion, epoch=1000, early_stop=3, scheduler=None):
    
    best_acc = 0
    for e in range(epoch):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        for i, (X, y, ind) in pbar:
            ############################################
            #TO DO
            #write a trainer to train your CNN model
            #evaluate your model on development set every epoch
            #you are expected to achieve between 0.50 to 0.70 accuracy on development set
            ############################################

            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(X)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())
            pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f}".format((e+1), np.mean(loss_log)))

        model.eval()
        logits = []
        ys = []
        for X,y,ind in dev_loader:
            logit = model(X)
            logits.append(logit.data.cpu().numpy())
            ys.append(y.data.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        preds = np.argmax(logits, axis=1)
        ys = np.concatenate(ys, axis=0)
        acc = accuracy_score(y_true=ys, y_pred=preds)
        label_names = ['rating 0', 'rating 1','rating 2']
        report = classification_report(ys, preds, digits=3,
                                    target_names=label_names)
        if acc>best_acc:
            best_acc=acc
        else:
            early_stop-=1
        print("current validation report")
        print("\n{}\n".format(report))
        print()
        print("epcoh: {}, current accuracy:{}, best accuracy:{}".format(e+1,acc,best_acc))
        
        if early_stop==0:
            break
        if scheduler is not None:
            scheduler.step()
    return model, best_acc

def predict(model, test_loader, save_file="submission.csv"):
    logits = []
    inds = []
    model.eval()
    for X,ind in test_loader:
        logit = model(X)
        logits.append(logit.data.cpu().numpy())
        inds.append(ind.data.cpu().numpy())
    logits = np.concatenate(logits, axis=0)
    inds = np.concatenate(inds, axis=0)
    preds = np.argmax(logits, axis=1)
    result = {'id':list(inds), "rating":preds}
    df = pd.DataFrame(result, index=result['id'])
    df.to_csv(save_file)  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--kernel_num", type=int, default=100)
    parser.add_argument("--kernel_sizes", type=str, default='3,4,5')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    args = parser.parse_args()
    #load data
    train_loader, dev_loader, test_loader, embedding_matrix = get_dataloaders(args.batch_size, args.max_len)
    #build model
    # try to use pretrained embedding here

    ## Word2Vector from Google
    ## https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    ## https://github.com/mmihaltz/word2vec-GoogleNews-vectors
    ## https://github.com/facebookresearch/fastText/issues/458
    ## 300 dimension

    model = WordCNN(args, embedding_matrix=embedding_matrix)
    #loss function
    criterion = nn.CrossEntropyLoss()
    #choose optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    model, best_acc = trainer(train_loader, dev_loader, model, optimizer, criterion, early_stop = args.early_stop)

    print('best_dev_acc:{}'.format(best_acc))
    predict(model, test_loader)

if __name__=="__main__":
    main()