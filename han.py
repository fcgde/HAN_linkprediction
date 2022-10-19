import torch
import torch.nn as nn
from model import NodeAttention, SemanticAttention, LayerNodeAttention
import torch.nn.functional as F

class HAN(nn.Module):

    def __init__(self, num_nodes, input_features, n_hid, head_number, meta_path_number, dropout=0.3, alpha=0.5, aloha = 0.5, link_prediction_layer = 0):
        super(HAN, self).__init__()

        #此处的参数传给__init__（）
        self.node_attention = NodeAttention(input_features, n_hid, head_number, dropout, alpha, meta_path_number, num_nodes)

        self.semantic_attention = SemanticAttention(n_hid, n_hid, aloha, link_prediction_layer)

        self.layer_node_attention = LayerNodeAttention(n_hid,n_hid, head_number, dropout, alpha, meta_path_number, num_nodes)

        self.dropout = dropout

        self.distmult = Dis(n_hid) # 此处的n_hid参数传给__init__（）

    # 也就是说继承nn.Module的类实例化后再传参即是传给forward（）

    def forward(self, features, adjs): #先节点级注意力再语义级注意力

        Z = self.node_attention(features, adjs) #此处的features, adjs传给forward

        Z = F.dropout(Z, self.dropout, training=self.training)
        #以上Z为三维矩阵Tensor(2,3025,64)

        # 层间节点特征矩阵计算start
        # Y = self.layer_node_attention(Z)
        # print(Y.shape)
        # print(Y)  # Y输出为（3025，2，64） 需要调整为（2，3025，64）
        # end

        Z = self.semantic_attention(Z) #传给forward的参数：特征矩阵     Z为三维矩阵Tensor(2,3025,64)
        # 语义级注意力处理后的 Z矩阵[node_number, features_number]

        Z = F.dropout(Z, self.dropout, training=self.training) #Z为三维矩阵Tensor(3025,64)

        score = self.distmult(Z) # 此处的Z参数传给forward（） Z矩阵[node_number,features_number]

        return score

class Dis(nn.Module):
    def __init__(self,f_dim,w_init='standard-normal',): # f_dim为HAN模型传入的n_hid，即训练输出的节点特征个数：outfeatures
        super(Dis, self).__init__()
        self.w_init = w_init

        '''将一个固定不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面
        经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。'''
        self.relations = nn.Parameter(torch.FloatTensor(f_dim)) # 获得tensor[f_dim,] f_dim即n_hid
        self.initialise_parameters()

    def initialise_parameters(self):
        # Weights
        # 按正态分布给矩阵随机赋值
        init = torch.nn.init.normal_
        init(self.relations)

    def compute_penalty(self):
        return self.relations.pow(2).sum() #pow(n)是对每个元素计算n次方，sum（）计算矩阵所有元素的和


    def forward(self, fea):# fea为特征矩阵[node_number,features_number]
        r = self.relations.repeat(fea.shape[0],1)
        #repeat第一个参数为重复relations所有元素的次数，第二参数整体广播relations所有元素，最终地矩阵[fea.shape[0],f_dim]
        scores = (fea * r).matmul(fea.T)
        # fea*r [node_number,feature_number]*[node_number,feature_number=f_dim=n_hid],是对应元素相乘
        penalty = self.compute_penalty()
        return scores , penalty


