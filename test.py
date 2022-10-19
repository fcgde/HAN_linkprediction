import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio

#a = torch.tensor([[1,2],[2,3]])
#b = torch.tensor([[4,5],[6,7]])
##for i,j in a,b:
#    print(i)
#   print(j)

##测试a，b
print('--------------------')
##print(a,b)
#print(a.data)

#查看.mat文件格式
'''key:dict_keys(['__header__', '__version__', '__globals__', 'PTP', 'PLP', 'PAP', 'feature', 'label', 'train_idx', 'val_idx', 'test_idx'])'''
'''value：矩阵'''
#data = scio.loadmat("./data/ACM3025.mat")
#print(data.items())
#print('--------------------')
#print("key",data.keys())
#print('--------------------')
#print("value",data.values())
#print("features：",data['feature'])
#print("label：",data['label'])
#print("PLP：",data['PLP'])
#print("PAP：",data['PAP'])
# checkShape = torch.Tensor(data['feature'])
# print('shapes:{:d},{:d}'.format(checkShape.shape[0],checkShape.shape[-1]))

# PLP=torch.Tensor(data["PLP"])
# print(PLP)

#查看break_text
# file = open('process_data/break.txt', mode='r', encoding='UTF-8')
# contents = file.read()
# contents = contents.split(' ')
# print(contents)
# cnt = 0
# pos = []
# while (cnt < len(contents) - 1):
#     k = []
#     k.append(int(contents[cnt]))
#     cnt += 1
#     k.append(int(contents[cnt]))
#     cnt += 1
#     pos.append(k)
# print(pos)

#查看train_idx
# file = open('process_data/train_idx.txt', mode='r', encoding='UTF-8')
# contents = file.read()
# contents = contents.split(' ')
# cnt = 0
# train_idx = []
# while (cnt < len(contents) - 1):
#     train_idx.append(int(contents[cnt]))
#     cnt += 1
# print(train_idx)

'''函数测试'''
# aa = torch.FloatTensor(64)
# #print(aa)
# testFuction = nn.Parameter(aa)
# #print(testFuction)
# result1 = testFuction.pow(2).sum()
# print("the result is:",result1)

# aa = torch.tensor([12,3,4,14,0.4,13,4,5])
# print(aa.gt(0.5))

# a = torch.tensor([[1,2],[2,3]])
# b = torch.tensor([[4,5],[6,7]])
# print(a.matmul(b))

#aa = nn.Parameter(torch.empty(255, 64)) #torch.empty():用来返回一个没有初始化的tensor
# bb = nn.init.xavier_uniform_(aa, 1.414)
# print(bb)
# aa = torch.empty(12,4)
# aa.reshape(aa.shape[0],aa.shape[1])
# print(aa)

#aa = torch.FloatTensor(64)
# init = torch.nn.init.normal_
# init(aa)
# bb = aa.pow(2).sum()
# print(bb)

#cc = torch.tensor(12,6)
# r = aa.repeat(24,2)
# print(aa.shape,r.shape)
# print(aa)
# print(r)


# aa = torch.tensor([
#     [1,2,3],
#     [4,3,2]
# ])
#
# bb = torch.tensor([
#     [0.166,0.166,0.166],
#     [0.166,0.166,0.166]
# ])
#print(aa.pow(2).sum())
# aa = torch.tensor([
#       [1,2,3],
#       [4,3,2]
#   ])
# bb = aa.reshape(-1)
# print(bb.shape[0])
# print(bb.shape[1])
#aa = list((1,1,0,0))
# aa = [[1,1,0,1]]
# bb = (1-aa)
# print(bb)


# aa = np.array(
#     [1,2,3,4,5,6,7]
# )
# bb = [3,1,5,8]
# cc = np.isin(aa,bb)
# print(cc)
# dd = (1-cc).astype(bool)
# print(dd)

#aa=nn.ModuleList(
    #[nn.init.xavier_uniform_(nn.Parameter(torch.empty(2, 2)), 1.414). for i in range(2)]
    #[nn.Linear(10, 10) for i in range(10)]
    #[nn.Linear(10, 10) for i in range(10)]
#)
#aa.add_module("trans1",[nn.Parameter(torch.empty(12, 12))])
#aa.add_module("trans2",[nn.Parameter(torch.empty(12, 12))])
#print(aa)
# print(aa[0])
# print(aa[1])

# a = torch.tensor([[1,2],[2,3]])
# print(a.shape[0])

# a = torch.tensor([
#         [
#             [1,2],
#             [2,3],
#             [3,4]
#         ],
#         [
#             [1,2],
#             [2,3],
#             [3,4]
#         ]
# ])
# print(a.shape)
# print(torch.transpose(a,2,1))
# b = torch.tensor(
#     [
#         [1],
#         [0]
#     ]
# )
# c = a.matmul(b)
# print(a.shape)
# print(b.shape)
# print(c.shape)
# d = c.reshape(c.shape[0], c.shape[1])
# print(c)
# print(d)

# a = torch.zeros(5,1)
# a[4,0] = 1
# print(a)
# b = torch.ones(5,5)
# c = b.matmul(a)
# print(c)

#print(len(a))

# a = torch.tensor([[1,2],
#      [3,4]])
# b = torch.tensor([
#     [1,2],
#     [3,4]
# ])
# c=torch.stack([a,b])
# print(c)
# print(c.shape)

# a = torch.ones(5,5)
# print(a)
# for i in range(10):
#     print(i)

if 3<2 :
    print("1")
else:
    print("2")