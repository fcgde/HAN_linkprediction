import torch
import numpy as np

a = torch.tensor([[1,2],[2,3]])
b = torch.tensor([[4,5],[6,7]])
for i,j in a,b:
    print(i)
    print(j)