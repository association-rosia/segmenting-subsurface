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


class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, labels, smooth=1):
        logits = logits.view(-1)
        outputs = F.sigmoid(logits)
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        dice = (2. * (outputs * labels).sum() + smooth) / (outputs.sum() + labels.sum() + smooth)

        pos_weight = self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        return 1 - dice + bce


class JaccardBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(JaccardBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, labels, smooth=1):
        logits = logits.view(-1)
        outputs = F.sigmoid(logits)
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        intersection = (outputs * labels).sum()
        jaccard = (intersection + smooth) / ((outputs + labels).sum() - intersection + smooth)

        pos_weight = self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        return jaccard + bce
