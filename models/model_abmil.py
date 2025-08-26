import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_clam import Attn_Net, Attn_Net_Gated

class ABMIL(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., n_classes=2, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        layers = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        layers.append(attention_net)
        self.attention_net = nn.Sequential(*layers)
        self.classifier = nn.Linear(size[1], n_classes)

    def forward(self, h, **kwargs):
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        logits = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat, A_raw, {}
