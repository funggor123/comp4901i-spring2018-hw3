import argparse
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, accuracy_score
from preprocess import get_dataloaders
from model import WordCNN
from main import trainer, predict


def tune(lr=0.1, dropout=0.3, kernel_num=100, kernel_sizes='3,4,5', embed_dim=100):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--dropout", type=float, default=dropout)
    parser.add_argument("--kernel_num", type=int, default=kernel_num)
    parser.add_argument("--kernel_sizes", type=str, default=kernel_sizes)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=embed_dim)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    args = parser.parse_args()

    # print("lr", args.lr, "dropout", args.dropout, "kernel_num", args.kernel_num, "kernel_sizes",args.kernel_sizes, "batch_size", args.batch_size, "early_stop", args.early_stop, "embed_dim", args.embed_dim, "max_len", args.max_len, "class_num", args.class_num, "lr_decay", args.lr_decay)
    train_loader, dev_loader, test_loader, vocab_size = get_dataloaders(args.batch_size, args.max_len)
    model = WordCNN(args, vocab_size, embedding_matrix=None)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # choose optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    model, best_acc = trainer(train_loader, dev_loader, model, optimizer, criterion, early_stop=args.early_stop)

    print('best_dev_acc:{}'.format(best_acc))
    predict(model, test_loader)
    print("This is args", args)


if __name__ == "__main__":
    # default 0.1
    lr = [0.1, 0.01]
    # default 0.3
    dropout = [0, 0.1, 0.3, 0.5]
    # default 100
    kernel_num = [50, 100, 150]
    # default 3,4,5
    kernel_sizes = ['2,3,4', '3,4,5', '4,5,6']
    # default = 100
    embed_dim = [50, 100, 200]
    # train with different learning rate
    for _lr in lr:
        print("Traing with different learning rate: " + str(_lr))
        tune(lr=_lr)
    for _dropout in dropout:
        print("Traing with different dropout: " + str(_dropout))
        tune(dropout=_dropout)
    for _kernel_num in kernel_num:
        print("Traing with different kernel_num: " + str(_kernel_num))
        tune(kernel_num=_kernel_num)
    for _kernel_size in kernel_sizes:
        print("Traing with different kernel_sizes: " + _kernel_size)
        tune(kernel_sizes=_kernel_size)
    for _embed_num in embed_dim:
        print("Traing with different embed_dim: " + str(_embed_num))
        tune(embed_dim=_embed_num)
