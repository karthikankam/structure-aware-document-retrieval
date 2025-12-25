import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.t = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, q, d):
        logits = q @ d.T / self.t
        labels = torch.arange(q.size(0), device=q.device)
        return (self.ce(logits, labels) + self.ce(logits.T, labels)) / 2
