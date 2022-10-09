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
    break_adj,test_idx, pos = break_link(adjs_list[id_layer], 0.1)

    # 取 剩下90% 的正样本和负样本 并打乱
    adj_2_path = adjs_list[id_layer] @ adjs_list[id_layer].T
    train_idx = get_sample(break_adj, test_idx, adj_2_path)

    # 取测试集 的负样本
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