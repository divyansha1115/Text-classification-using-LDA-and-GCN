import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class SGC(nn.Module):
    def __init__(self, nfeat, nclass, bias=False):
        super(SGC, self).__init__()
        self.nclass = nclass
        self.W = nn.Linear(nfeat, nclass, bias=bias)
        torch.nn.init.xavier_normal_(self.W.weight)
        # 2 for mr else 2 * class
        self.dense = nn.Linear(2*nclass, nclass, bias=bias)
        self.dense1 = nn.Linear(nclass, nclass, bias=bias)
        self.dropout = nn.Dropout(0.9)
        self.bn1 = nn.BatchNorm1d(num_features=46)
        self.sm = nn.Softmax()

    def gather(self, y):
        leng = y.size(1) / 2
        y = y.permute((1, 0))
        new_tensor = []
        for i in range(int(leng)):
            new_tensor.append(y[:][i])
            new_tensor.append(y[:][i+int(leng)])
        return torch.stack(new_tensor).transpose(0, 1)

    def forward(self, x, lda_features):
        # print(x.size(), lda_features.size())
        # print(self.nclass)
        out = self.W(x)
        # print(out.size())
        # out = self.sm(out)
        out = out.float()
        # lda_features = self.sm(lda_features)
        lda_features = lda_features.float()
        new_out = torch.cat((out, lda_features), dim=1)
        new_out = self.gather(new_out)
        # new_out = self.bn1(new_out)

        new_out = self.dropout(new_out)
        new_out = self.dense(new_out)
        # new_out = self.dropout(new_out)
        new_out = new_out + out

        new_out1 = self.dense1(new_out)
        new_out1 = self.dropout(new_out1)
        new_out = new_out + new_out1

        new_out1 = self.dense1(new_out)
        new_out1 = self.dropout(new_out1)
        new_out = new_out + new_out1

        new_out1 = self.dense1(new_out)
        new_out1 = self.dropout(new_out1)
        new_out = new_out + new_out1


        return new_out
