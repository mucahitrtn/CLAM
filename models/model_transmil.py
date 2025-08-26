import torch
import torch.nn as nn
import torch.nn.functional as F

class TransMIL(nn.Module):
    def __init__(self, n_classes=2, input_dim=1024, embed_dim=256, n_heads=4, n_layers=2, dropout=0.25):
        super().__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, h, **kwargs):
        x = self.fc(h)  # N x embed_dim
        cls_tok = self.cls_token
        x = torch.cat([cls_tok.squeeze(1), x], dim=0)  # (N+1) x embed_dim
        x = x.unsqueeze(1)  # (N+1) x 1 x embed_dim
        x = self.transformer(x)
        cls_feat = x[0]
        logits = self.classifier(cls_feat)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat, None, {}
