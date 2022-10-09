import numpy as np
import torch
import scipy.io as scio
import pandas as pd

id_layer = 'PAP'   # 要预测第几层

def load_data():
    data = scio.loadmat("../data/ACM3025.mat")

    adjs = {
        "PLP": torch.Tensor(data["PLP"]),
        "PAP": torch.Tensor(data["PAP"])
    }
    # 上述加载的是两条元路径PAP和（应该是PSP的，不知道这里为何其键值为PLP）PSP的邻接矩阵

    origin_adj = adjs[id_layer]
    break_adj = adjs[id_layer]

    metapaths = adjs.keys()
    features = torch.Tensor(data["feature"])

    node_num = features.shape[0]

    # 读取break后的矩阵
    file = open('process_data/break.txt', mode='r', encoding='UTF-8')
    contents = file.read()
    contents = contents.split(' ')
    cnt = 0
    pos = []
    while (cnt < len(contents) - 1):
        k = []
        k.append(int(contents[cnt]))
        cnt += 1
        k.append(int(contents[cnt]))
        cnt += 1
        pos.append(k)
    for i in range(len(pos)):
        break_adj[pos[i][0]][pos[i][1]] = break_adj[pos[i][1]][pos[i][0]] = 0

    adjs[id_layer] = break_adj

    # 读取train_idx 并构建label
    file = open('process_data/train_idx.txt', mode='r', encoding='UTF-8')
    contents = file.read()
    contents = contents.split(' ')
    cnt = 0
    train_idx = []
    while (cnt < len(contents) - 1):
        train_idx.append(int(contents[cnt]))
        cnt += 1
    train_label = []
    for i in range(len(train_idx)):
        r = train_idx[i] // node_num
        c = train_idx[i] % node_num
        train_label.append(origin_adj[r][c])

    # test
    file = open('process_data/test_idx.txt', mode='r', encoding='UTF-8')
    contents = file.read()
    contents = contents.split(' ')
    cnt = 0
    test_idx = []
    while (cnt < len(contents) - 1):
        test_idx.append(int(contents[cnt]))
        cnt += 1
    test_label = []
    for i in range(len(test_idx)):
        r = test_idx[i] // node_num
        c = test_idx[i] % node_num
        test_label.append(origin_adj[r][c])

    return features, adjs, break_adj, metapaths, train_idx,test_idx, train_label, test_label

def process_load_data():
    data = scio.loadmat("../data/ACM3025.mat")

    adjs = {
        "PLP": torch.Tensor(data["PLP"]),
        "PAP": torch.Tensor(data["PAP"])
    }
    # 上述加载的是两条元路径PAP和（应该是PSP的，不知道这里为何其键值为PLP）PSP的邻接矩阵

    metapaths = adjs.keys()
    features = torch.Tensor(data["feature"])
    # label = torch.argmax(torch.Tensor(data["label"]), dim=1)
    train = data["train_idx"].reshape(-1)
    val = data["val_idx"].reshape(-1)
    test = data["test_idx"].reshape(-1)

    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    return features, adjs, metapaths

if __name__ == "__main__":
    file = open('process_data/test_idx.txt', mode='r', encoding='UTF-8')
    contents = file.read()
    contents = contents.split(' ')
    cnt = 0
    test_idx = []
    while (cnt < len(contents) - 1):
        test_idx.append(int(contents[cnt]))
        cnt += 1
    print(len(test_idx))
