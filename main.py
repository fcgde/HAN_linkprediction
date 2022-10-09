import torch
import numpy as np
import scipy.io as scio
import torch.nn as nn
import torch.nn.functional as F
from han import *
from utils import *
from model import *
from sample import *
from torch.utils.tensorboard import SummaryWriter

# 添加tensorboard
# writer = SummaryWriter("./logs2/dir2")
# writer = SummaryWriter("./logs2/dir5")


if __name__ == "__main__":
    features, adjs, break_adj, metapaths, train_idx,test_idx, train_label, test_label = load_data()

    print("训练集样本个数：",len(train_idx))
    print("测试集样本个数：",len(test_idx))

    num_nodes = features.shape[0]
    dim_feature = features.shape[1]
    print("num_nodes:{:d}".format(num_nodes))
    print("dim_features:{:d}".format(dim_feature))
    model = HAN(input_features=dim_feature, n_hid=64,head_number=4,  meta_path_number=2, aloha = 0.6, link_prediction_layer=1)
    device = torch.device("cuda")
    model = model.to(device)
    features = features.to(device)

    lr = 0.01
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0)
    total_epochs = 500

    lam = 0.001

    train_label = torch.tensor(train_label).to(device)
    test_label = torch.tensor(test_label).to(device)

    for epoch in range(total_epochs):

        model.train()
        outputs , penalty = model(features, adjs)
        outputs = torch.flatten(outputs)
        outputs_train = outputs[train_idx]

        loss = nn.functional.binary_cross_entropy_with_logits(outputs_train,train_label.float())
        loss = loss + (lam * penalty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs_train_r = outputs_train.gt(0.5)

        correct = torch.sum(outputs_train_r == train_label).to(torch.float32)
        acc = correct / len(train_label)
        print("epoch:{:d}  train_loss:{:f}  train_acc:{:f}".format(epoch,loss,acc))

        # writer.add_scalar("train_loss", loss, epoch)

        model.eval()
        with torch.no_grad():
            outputs_test = outputs[test_idx]

            loss = nn.functional.binary_cross_entropy_with_logits(outputs_test, test_label.float())
            loss = loss + (lam * penalty)

            outputs_test_r = outputs_test.gt(0.5)


            correct = torch.sum(outputs_test_r == test_label).to(torch.float32)

            acc = correct/ len(test_label)
            print("epoch:{:d}  test_loss:{:f}  test_acc:{:f}".format(epoch, loss, acc))

            # if epoch >=10:
            #     writer.add_scalar("test_acc", acc, epoch-10)

# writer.close()