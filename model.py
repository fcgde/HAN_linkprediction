import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class NodeAttentionPerMetaPath神经网络接收parameter：特征矩阵x和该元路径下的邻接矩阵
returan: 训练输出特征矩阵[node_number,out_features // head_number]
引入处有操作：output_features // head_number  此时的特征是一个head下的特征矩阵，返回出后会进行列扩维拼接
'''
class NodeAttentionPerMetaPath(nn.Module):
    """
    This class will implement the node-level attention for one meta-path
    """

    def __init__(self, input_features, output_features, dropout, alpha, adition_tag): # adition_tag判断层内还是层间
        super(NodeAttentionPerMetaPath, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.dropout = dropout
        self.alpha = alpha

        self.leakyReLU = nn.LeakyReLU(alpha) #LeakyReLU激活函数

        '''nn.Parameter():其作用将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数，
        并将这个参数绑定到module里面，成为module中可训练的参数。'''
        self.trans1 = nn.Parameter(torch.empty(input_features, output_features)) #torch.empty():用来返回一个没有初始化的tensor
        nn.init.xavier_uniform_(self.trans1.data, 1.414) #torch.nn.init.xavier_uniform_是一个服从均匀分布的Glorot初始化器

        self.trans2 = nn.Parameter(torch.empty(output_features, output_features))  # torch.empty():用来返回一个没有初始化的tensor
        nn.init.xavier_uniform_(self.trans2.data, 1.414)

        self.trans3 = nn.Parameter(torch.empty(adition_tag, output_features))  # torch.empty():用来返回一个没有初始化的tensor
        nn.init.xavier_uniform_(self.trans2.data, 1.414)


        self.attention = nn.Parameter(torch.empty(2 * output_features, 1))
        nn.init.xavier_uniform_(self.attention.data, 1.414)

        #self.trans , self.attention随机生成，后续自学习更新

    def forward(self, x, mask):
        # mask is the adjacent matrix for this meta-path
        x = F.dropout(x, self.dropout, training=self.training)
        '''F.dropout 是一个函数，参数包含输入的tensor，概率和training 为真还是假。当training 是真的时候，才会将一部分元素置为0，
        其他元素会乘以 scale 1/(1-p). training 为false的时候，dropout不起作用。默认情况下training是True'''

        x = x.matmul(self.trans1)
        #[node_number,input_features] x [input_features,output_features] = [node_number,output_features]
        # transform into [node_number,output_features]

        e_1 = x.matmul(self.attention[:self.output_features, :]) #切片
        e_2 = x.matmul(self.attention[self.output_features:, :]) #切片
        #e_1和e_2公式  [node_number,output_features] x [output_features,1] = [node_number,1] 即每个节点的重要性

        scores = self.leakyReLU(e_1 + e_2.T) # e_1 + e_2.T -> [node_number,1] + [1,node_number] 广播= [node_number,node_number]
        scores = F.dropout(scores, self.dropout, training=self.training)

        mask = mask.to(device=torch.device("cpu"))
        masked_scores = scores.masked_fill_(mask == 0, -1e15) # masked_fill_(mask, value) mask是张量，元素是布尔值， value是要填充的值
        #邻接矩阵中为0的元素对应得分矩阵scores设置为0
        attention = F.softmax(masked_scores, dim=1)
        Z = attention.matmul(x)
        # 二次卷积位置
        # masked_scores = self.leakyReLU(masked_scores)???

        # masked_scores = F.dropout(masked_scores, self.dropout, training=self.training)???

        #判断是层间还是层内
        if Z.shape[0] == 3025 :
            x = Z.matmul(self.trans2)
        else:
            x = Z.matmul(self.trans3)
        e_3 = x.matmul(self.attention[:self.output_features, :])
        e_4 = x.matmul(self.attention[self.output_features:, :])
        scores = self.leakyReLU(e_3 + e_4.T)
        scores = F.dropout(scores, self.dropout, training=self.training)
        masked_scores = scores.masked_fill_(mask == 0, -1e15)
        #
        #
        #
        # '''根据不同的dim规则来做归一化操作。x指的是输入的张量，dim指的是归一化的方式。
        # dim=0,按列SoftMax，列和为1（即0维度进行归一化）
        # dim=1,按列SoftMax，列和为1（即1维度进行归一化）'''
        attention = F.softmax(masked_scores, dim=1)

        #[node_number,node_number] x [node_number,output_features] = [node_number,output_features]
        # 即某一结点对其他节点的注意力值 x 每个节点的某一特征（特征1->特征output_features） = 该节点的输出特征
        return attention.matmul(x)


        # matmul():矩阵乘积，类似于矩阵相乘操作的tensor连乘操作。
        # 但是它可以利用python中的广播机制，处理一些维度不同的tensor结构进行相乘操作

    #不考虑node-level的attention

    # def forward(self, x, mask):
    #     # mask is the adjacent matrix for this meta-path
    #     x = F.dropout(x, self.dropout, training=self.training)
    #
    #     x = x.matmul(self.trans)
    #     # transform into [node_number,output_features]
    #
    #     scores = torch.ones(3025,3025)
    #     scores = scores.to(device=torch.device("cuda"))
    #
    #     # scores = F.dropout(scores, self.dropout, training=self.training)
    #
    #     mask = mask.to(device=torch.device("cuda"))
    #     masked_scores = scores.masked_fill_(mask == 0, -1e15)
    #
    #     attention = F.softmax(masked_scores, dim=1)
    #
    #     return attention.matmul(x)


'''
class MultiHeadNodeAttentionPerMetaPath 接收parameter:特征矩阵x和该元路径下的邻接矩阵
该类的功能对特征值分组训练output_features // head_number 
功能函数：NodeAttentionPerMetaPath(input_features, output_features // head_number, dropout, alpha)和attention(x, mask)
return：训练输出特征矩阵[node_number,out_features]
'''
class MultiHeadNodeAttentionPerMetaPath(nn.Module):

    def __init__(self, input_features, output_features, dropout, alpha, head_number, adition_tag): # adition_tag判断层内还是层间
        super(MultiHeadNodeAttentionPerMetaPath, self).__init__()

        assert output_features % head_number == 0
        # make sure that features number for each head is an integer

        self.attentions = nn.ModuleList(
            [NodeAttentionPerMetaPath(input_features, output_features // head_number, dropout, alpha, adition_tag)
             for _ in range(head_number)])

    def forward(self, x, mask): # x为han.py传的features，mask为该元路径的邻接矩阵adjs[meta-path]
        return torch.cat([attention(x, mask) for attention in self.attentions], dim=1) #对每一head_number下训练出来的节点特征矩阵拼接
        #torch.cat为拼接函数 outputs = torch.cat(inputs, dim=?) → Tensor
        # inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
        # dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列  dim=1为列扩维

'''
class: 节点级注意力
'''
class NodeAttention(nn.Module):

    def __init__(self, input_features, output_features, head_number, dropout, alpha, metapath_number, addition_tag):
        super(NodeAttention, self).__init__()

        self.metapaths_attentions = nn.ModuleList( #nn.ModuleList,它是一个存储不同module，并自动将每个module的parameters添加到网络之中的容器
            #nn.ModuleList并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言
            [MultiHeadNodeAttentionPerMetaPath(input_features, output_features, dropout, alpha, head_number, addition_tag)
             for _ in range(metapath_number)])

        self.metapath_number = metapath_number

    def forward(self, x, adjs): # x为han.py传的features，adjs为所有元路径的邻接矩阵
        # adjs is the mask for different meta path

        assert len(adjs) == self.metapath_number

        # return a [metapath_number,node_number,out_features] tensor
        tmp_arr = []
        # tmp_arr.append(0)
        for item in adjs:
            tmp_arr.append(adjs[item]) # 获取每一元路径的邻接矩阵
        # print(tmp_arr)

        # torch.stack()对tensors沿指定维度拼接，但返回的Tensor会多一维。。torch.cat()返回的Tensor维度不变
        #也即第一维为元路径，每一元路径又对应训练返回的特征矩阵
        return torch.sigmoid(torch.stack( # 循环元路径，并给metapaths_attentions的forward传参
            [meta_path_attention(x, tmp_arr[i]) for i, meta_path_attention in enumerate(self.metapaths_attentions)]))  #对每一元路径训练返回相应特征矩阵


'''class: 语义级注意力
'''
class SemanticAttention(nn.Module):

    def __init__(self, input_features, output_features,aloha,link_prediction_layer):
        super(SemanticAttention, self).__init__()

        #  W为权值矩阵[input_features, output_features]
        self.W = nn.Parameter(torch.empty(input_features, output_features))#torch.empty():用来返回一个没有初始化的tensor
        nn.init.xavier_uniform_(self.W.data, 1.414) #torch.nn.init.xavier_uniform_是一个服从均匀分布的Glorot初始化器

        # b为偏置向量[1, output_features]
        self.b = nn.Parameter(torch.empty(1, output_features))
        nn.init.xavier_uniform_(self.b.data, 1.414)

        # q是语义级别的attention向量[output_features, 1]
        self.q = nn.Parameter(torch.empty(output_features, 1))
        nn.init.xavier_uniform_(self.q.data, 1.414)
        # 分别对应文章当中的W，b和q

        # 阿尔法参数
        self.aloha = aloha

        # 预测第几层的，该层乘以阿尔法
        self.link_prediction_layer = link_prediction_layer

    def forward(self, node_attentions):  #此处node_attentions=Tensor(2,3025,64)
        # input is a [meta_path_number,node_number,input_features] tensor 特征矩阵

        trans = torch.tanh(node_attentions.matmul(self.W) + self.b)  #此处node_attentions=Tensor(2,3025,64)
        # trans is a [meta_path_number,node_number,output_features] tensor
        # trans=Tensor(2,3025,64)

        # 此处trans乘q后trans（2，3025，1）,reshape后w_meta=Tensor(2,3025)
        w_meta = trans.matmul(self.q).reshape(trans.shape[0], trans.shape[1])

        # w_meta is a [meta_path_number,node_number] tensor

        w_meta = w_meta.sum(dim=1) / w_meta.shape[1]
        # 计算注意力的分数,得到每一元路径给的分数

        beta = F.softmax(w_meta, dim=-1) #对元路径维归一化
        # 计算softmax之后的结果

        Z = torch.zeros(size=node_attentions[0].shape).to(torch.device("cpu")) # 矩阵[meta_path_number,1]

        for i, weight in enumerate(beta): # 对每一元路径循环计算 i和weight
            if(i == self.link_prediction_layer):
                Z += self.aloha * node_attentions[i] # node_attentions[i]为元路径i下的经节点级注意力训练输出特征矩阵
            else:
                Z += (1 - self.aloha) * weight * node_attentions[i]
        # 加权求和
        return Z

'''
层间节点权重计算
'''
class LayerNodeAttention(nn.Module):
    def __init__(self, input_features, output_features, head_number, dropout, alpha, metapath_number, num_nodes):
        super(LayerNodeAttention, self).__init__()
        self.metapaths_attentions = nn.ModuleList(  # nn.ModuleList,它是一个存储不同module，并自动将每个module的parameters添加到网络之中的容器
            # nn.ModuleList并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言
            [MultiHeadNodeAttentionPerMetaPath(input_features, output_features, dropout, alpha, head_number ,metapath_number)
             for _ in range(num_nodes)])
        self.input_features = input_features
        self.metapath_number = metapath_number


    '''计算层间节点的特征矩阵
    node_features为层内节点注意力后的特征矩阵，三维
    i为第i个节点
    '''
    def layer_node_attention(self, node_features, i):
        a = torch.zeros(node_features.shape[1], 1)  # 要得到i节点的层间特征矩阵就设第几行为1，其余为0
        a[i,0] = 1
        layer_attention = torch.transpose(node_features,2,1)
        b = layer_attention.matmul(a).reshape(layer_attention.shape[0], layer_attention.shape[1])
        # b为第i个节点的层间特征矩阵 = Tensor(meta_path, out_features)
        return b


    def forward(self, node_features):
        # node_features=Tensor(meta_path,node_number,out_features)为层内节点级注意力后的特征矩阵
        tmp_arr = []
        # tmp_arr.append(0)
        for item in range(node_features.shape[1]):
            tmp_arr.append(torch.ones(self.metapath_number,self.metapath_number))  # 获取每一节点的层间邻接矩阵,组成列表


        layer_all_attention = torch.stack([
            self.layer_node_attention(node_features, i) for i in range(node_features.shape[1])
        ]) # 得到层间每个节点的特征矩阵，Tensor=（node_number, meta_path, out_feaures）

        # 下一步计算层间节点的权重矩阵
        Z = torch.stack( #思路：把每一个节点的层间特征矩阵和每个节点的层间邻接矩阵传参，
            # 循环拼接函数返回的每个节点的层间特征矩阵，返回特征矩阵是经过层间节点权重处理过的
            [meta_path_attention(layer_all_attention[i], tmp_arr[i]) for i, meta_path_attention in
             enumerate(self.metapaths_attentions)]) # 对层间每一节点训练返回相应特征矩阵
        # Z = Tensor（node_number, meta_path, out_features）

        Z = torch.transpose(Z,1,2) # 转置Z -> Tensor(node_number,out_features,meta_path)
        Y = torch.sigmoid(torch.stack([
            self.transZshape(Z, Z.shape[2], i) for i in range(Z.shape[2])
        ])) # Y=Tensor(meta_path,node_number,outfeatures)

        return Y

    def transZshape(self, z, dim, i):
        # z=Tensor(node_number,meta_path,out_featurs)
        # matrics为单位列矩阵，获取第几层节点的层间特征矩阵，单位矩阵第几行就为1
        matrics = torch.zeros(dim, 1)
        matrics[i, 0] = 1
        m = z.matmul(matrics).reshape(z.shape[0], z.shape[1])  # m=Tensor(node_number,out_features)
        return m # m为第i层节点的层间特征矩阵

