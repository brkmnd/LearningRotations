import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LearningRotationNN(nn.Module):
    def __init__(self,input_dim,inter_dim,labels_dim):
        super(LearningRotationNN,self).__init__()

        self.lin1 = nn.Linear(input_dim,inter_dim)
        self.lin2 = nn.Linear(inter_dim,labels_dim)

    def forward(self,x):
        out = self.lin1(x.view(1,-1))
        out = F.relu(out)
        out = self.lin2(out)
        return F.log_softmax(out,dim=1)
