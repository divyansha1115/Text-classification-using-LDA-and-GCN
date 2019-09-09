class SGC(nn.Module):
    def _init_(self, nfeat, nclass, bias=False):
        super(SGC, self)._init_()
        self.W = nn.Linear(nfeat, nclass, bias=bias)
        torch.nn.init.xavier_normal_(self.W.weight)
        self.dense = nn.Linear(46, nclass, bias=bias)
        self.dense1 = nn.Linear(23, nclass, bias=bias)
        self.dropout = nn.Dropout(0.9)
        self.bn1 = nn.BatchNorm1d(num_features=46)
        print(nfeat)
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
        out = self.W(x)
        # out = self.sm(out)
        out = out.float()
        # lda_features = self.sm(lda_features)
        lda_features = lda_features.float()
        new_out = torch.cat((out, lda_features), dim=1)
        print(new_out.size())
        new_out = self.gather(new_out)
        print(new_out.size())
        # new_out = self.bn1(new_out)
        # new_out = self.dropout(new_out)
        new_out = self.dense(new_out)
        new_out = self.dropout(new_out)
        new_out = new_out + out
        new_out1 = self.dense1(new_out)
        new_out1 = self.dropout(new_out1)
        new_out = new_out + new_out1
        return new_out