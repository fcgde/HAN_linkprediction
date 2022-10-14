import numpy as np
import torch
import scipy.io as scio
import pandas as pd

id_layer = 'PAP'   # 要预测第几层

def load_data():  #功能：获取features, adjs, break_adj, metapaths, train_idx,test_idx, train_label, test_label
    data = scio.loadmat("./data/ACM3025.mat")

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
    #将文本数字排列成（n，2）矩阵pos，借助矩阵pos打乱矩阵break_adj
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
    #将train_idx.txt整理成（1，n）矩阵train_idx，再将其元素整除和求余node_num，得到随机效果r，c
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
    #features为data["feature"]的矩阵
    #adjs为PAP,PLP元路径的矩阵{PAP,PLP}
    #break_adj为元路径PAP打乱后的结果矩阵
    #metapaths数据data的keys
    #train_idx为train_idx.txt形成的（1，n）矩阵
    #test_idx为test_idx.txt形成的（1，n）矩阵
    #train_label为长度len(train_idx)，随机获取origin_adj[r][c]得到的（1，n）矩阵
    #test_label为长度len(test_idx)，随机获取origin_adj[r][c]得到的（1，n）矩阵


def process_load_data():
    data = scio.loadmat("./data/ACM3025.mat")

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
