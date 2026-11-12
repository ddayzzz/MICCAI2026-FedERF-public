from torch import nn


class Classifier(nn.Module):
    def __init__(self, hs, cls_num):
        super().__init__()
        self.fc1 = nn.Linear(hs, hs * 2)
        self.ln1 = nn.LayerNorm(hs * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hs * 2, cls_num)

    def forward_hidden(self, x):
        out = self.fc1(x)
        out = self.ln1(out)
        z = self.act(out)
        return z

    def forward(self, z):
        hid = self.forward_hidden(z)
        return self.fc2(hid)