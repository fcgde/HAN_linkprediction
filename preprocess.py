from sample import *
from utils import *

id_layer = 1   # 要预测第几层

if __name__ == "__main__":
    features, adjs, metapaths = process_load_data()

    # 获得邻接矩阵list
    adjs_list = []
    for k,v in adjs.items():
        adjs_list.append(v)

    # 抹掉 PLP图中 10% 的边，并获取测试集的正样本
    # break_adj为原邻接矩阵破坏边后的矩阵，test_idx为抹去的边的存储位置，pos为破坏边的邻接矩阵位置
    break_adj,test_idx, pos = break_link(adjs_list[id_layer], 0.1)

    # 取 剩下90% 的正样本和负样本 并打乱
    adj_2_path = adjs_list[id_layer] @ adjs_list[id_layer].T # @运算符和numpy的matmul是一样的
    # 2跳矩阵adj_2_path[i,j]:节点i到j经过2条路径的数量
    train_idx = get_sample(break_adj, test_idx, adj_2_path)

    # 取测试集 的负样本
    # 抹去边的位置集和（test_idx）数量的原adj中为0的元素不在训练集且（2跳为0）的元素位置
    test_idx = test_negative_sampling(adjs_list[id_layer], test_idx, train_idx, adj_2_path)

    # 保存抹掉 10% 的边的位置 和训练、测试集的索引

    with open('process_data/break.txt','w') as f:
        for i in pos:
            f.write(str(i))
            f.write(' ')
        f.close()

    with open('process_data/train_idx.txt','w') as f:
        for i in train_idx:
            f.write(str(i))
            f.write(' ')
        f.close()

    with open('process_data/test_idx.txt','w') as f:
        for i in test_idx:
            f.write(str(i))
            f.write(' ')
        f.close()