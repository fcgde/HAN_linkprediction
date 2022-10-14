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

    num_nodes = features.shape[0] #节点数
    dim_feature = features.shape[1]#节点的特征值个数
    print("num_nodes:{:d}".format(num_nodes))
    print("dim_features:{:d}".format(dim_feature))
    model = HAN(input_features=dim_feature, n_hid=64,head_number=4,  meta_path_number=2, aloha = 0.6, link_prediction_layer=1)
    ##device = torch.device("cuda")
    device = torch.device("cpu")
    model = model.to(device)
    features = features.to(device)

    lr = 0.01
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0)
    # torch.optim.Adam()
    # params(iterable) – 待优化参数的iterable或者是定义了参数组的dict
    # lr(float, 可选) – 学习率（默认：1e-3）
    # betas(Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
    # eps(float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
    # weight_decay(float, 可选) – 权重衰减（L2惩罚）（默认: 0）

    total_epochs = 500  #epochs指的就是训练过程中全部样本数据将被“轮”多少次

    lam = 0.001

    train_label = torch.tensor(train_label).to(device)
    test_label = torch.tensor(test_label).to(device)

    #数据epoch最大500次
    for epoch in range(total_epochs):

        model.train() #将特殊层设置为训练模式
        outputs , penalty = model(features, adjs) #返回dismult的得分score和惩罚值penalty
        outputs = torch.flatten(outputs) #torch.flatten(t, start_dim=0, end_dim=-1)默认 start_dim=0，end_dim=-1
        outputs_train = outputs[train_idx]
        #train_idx提供坐标获取一定数量的outputs中的一些值 outputs：9150625 train_idx：52942 outputs_train：52942

        #function：对神经网络的输出结果进行sigmoid操作，然后求交叉熵
        '''input神经网络预测结果（未经过sigmoid）, 任意形状的tensor
        target标签值，与input形状相同
        weight权重值，可用于mask的作用， 具体作用下面讲解，形状同input
        size_average弃用，见reduction参数
        reduce弃用，见reduction参数
        reduction指定对输出结果的操作，可以设定为none mean sum;
        none将不对结果进行任何处理，mean对结果求均值， sum对结果求和， 默认是mean'''
        loss = nn.functional.binary_cross_entropy_with_logits(outputs_train,train_label.float())

        loss = loss + (lam * penalty)
        optimizer.zero_grad() #reset gradient 清空过往梯度
        loss.backward() #反向传播，计算当前梯度
        optimizer.step() #根据梯度更新网络参数

        outputs_train_r = outputs_train.gt(0.5) # 对矩阵元素 > 0.5 判断，返回true或false

        correct = torch.sum(outputs_train_r == train_label).to(torch.float32) #训练结果拟合的个数
        acc = correct / len(train_label)
        print("epoch:{:d}  train_loss:{:f}  train_acc:{:f}".format(epoch,loss,acc))

        # writer.add_scalar("train_loss", loss, epoch)

        model.eval() #开启预测模式 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
        with torch.no_grad(): # 无梯度下降
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