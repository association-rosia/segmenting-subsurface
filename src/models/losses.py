import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, labels, smooth=1):
        outputs = F.sigmoid(logits)
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        intersection = (outputs * labels).sum()
        dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)

        return 1 - dice


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, logits, labels, smooth=1):
        outputs = F.sigmoid(logits)

        dice = (2. * (outputs * labels).sum() + smooth) / (outputs.sum() + labels.sum() + smooth)

        weight = self.weight.to(logits.device)
        cross_entropy = F.cross_entropy(logits, labels, weight=weight)

        return 1 - dice + cross_entropy


class JaccardCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(JaccardCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, logits, labels, smooth=1):
        outputs = F.sigmoid(logits)

        intersection = (outputs * labels).sum()
        jaccard = (intersection + smooth) / ((outputs + labels).sum() - intersection + smooth)

        weight = self.weight.to(logits.device)
        cross_entropy = F.cross_entropy(logits, labels, weight=weight)

        return jaccard + cross_entropy
