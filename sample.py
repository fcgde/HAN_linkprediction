import numpy as np
import random
import math
import torch

def break_link(adj,break_portion):
    idx_test = []
    pos = []
    n = adj.shape[0] #代表节点数
    break_num = math.ceil(break_portion * torch.sum(adj) / 2)   # 对称的矩阵，断一半的边另一半也断
    cnt = 0
    while cnt < int(break_num):
        x_cor = random.randint(0, n - 1)
        y_cor = random.randint(0, n - 1)
        pos.append(x_cor)
        pos.append(y_cor)
        if adj[x_cor, y_cor] == 1 and torch.sum(adj[x_cor, :]) != 1 and torch.sum(adj[y_cor, :]) != 1: # 该位置的元素所在列、行元素和不为1
            idx_test.extend([x_cor*n + y_cor,y_cor*n + x_cor]) #idx_test为（testNode_number,2），因此expend（）需要计算抹点元素的位置
            adj[x_cor, y_cor] = adj[y_cor, x_cor] = 0
            cnt += 1
    return adj,idx_test, pos

def get_sample(adj, idx_test, adj_2_path):          # 获取训练集的正负样本  adj为break后的 adj_2_path 为break前的二跳
    idx_train_positive = np.array(list(np.where(np.array(adj).flatten() !=0))[0])   # 正样本的索引  输出为1维
    train_positive_num = idx_train_positive.shape[0] #正样本中的节点数
    zero_location = list(np.where(np.array(adj).flatten() == 0))[0]         # 为0的位置
    temp = np.isin(zero_location, idx_test)                 # 得到0的位置中 是否为测试数据的标记
    temp = (1-temp).astype(bool)  # 该步操作后temp矩阵可根据布尔值记录每个元素是否为idx_test中的元素

    zero_location_2_path = list(np.where(np.array(adj_2_path).flatten() == 0))[0]  # 2跳为0的位置

    temp2 = np.isin(zero_location, zero_location_2_path) #得到0的位置中 是否为 2跳为0的位置

    temp3 = temp & temp2 # adj中为0的元素不在测试集且0元素为2跳

    # 此处负样本idx_train_negative定义为邻接矩阵adj中为0且（adj中为0的元素不在测试集且0元素为2跳）
    idx_train_negative = np.random.choice(zero_location[np.where(temp3 == True)], size = train_positive_num, replace=False) # 选负样本的索引 输出为一维
    # np.vstack()按垂直方向（行顺序）堆叠数组构成一个新的数组
    # np.hstack()按水平方向（列顺序）堆叠数组构成一个新的数组
    idx_train = np.hstack((idx_train_negative, idx_train_positive))  # 正样本和负样本 输出为1维
    np.random.shuffle(idx_train)
    print('train negative sampling done')
    return idx_train

def test_negative_sampling(adj,idx_test_positive,idx_train, adj_2_path):  # 获取测试集的负样本  adj为break前的
    idx_test_positive = np.array(idx_test_positive)     # 正样本的索引
    test_positive_num = idx_test_positive.shape[0]
    zero_location = list(np.where(np.array(adj).flatten() == 0))[0]  #原邻接矩阵中元素为0的位置数组
    choice_pos = np.isin(zero_location,idx_train)
    choice_pos = (1-choice_pos).astype(bool)  # 不能在训练集中

    zero_location_2_path = list(np.where(np.array(adj_2_path).flatten() == 0))[0]  # 2跳为0的位置
    temp = np.isin(zero_location,zero_location_2_path) #判断原邻接矩阵中元素为0的位置数组在2跳为0的位置

    temp2 = choice_pos & temp #一维，原adj中为0的元素不在训练集且（2跳为0）的元素位置

    # 在temp2中随机选取test_positive_num数量
    idx_test_negative = np.random.choice(zero_location[np.where(temp2 == True)], size = test_positive_num, replace=False)
    # np.hstack()按水平方向（列顺序）堆叠数组构成一个新的数组
    idx_test = np.hstack((idx_test_positive, idx_test_negative ))
    np.random.shuffle(idx_test)
    return idx_test

# def get_batch(train_idx,train_label,batch_size):
#     train_idx_bt = []
#     train_label_bt = []
#     l = len(train_idx)
#     cnt = 0
#     while(cnt * batch_size < l):
#         id = []
#         lab = []
#         for i in range(batch_size):
#             id.append(train_idx[i])
#             lab.append(train_label[i])
#         train_idx_bt.append(id)
#         train_label_bt.append(lab)
#         cnt += 1
#     return train_idx_bt,train_label_bt,cnt