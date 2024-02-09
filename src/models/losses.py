import torch.nn.functional as tF
import torchmetrics.functional as tmF
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_labels=None, class_weights=None):
        super(CrossEntropyLoss, self).__init__()
        self.num_labels = num_labels
        self.class_weights = class_weights

    def forward(self, input, target):
        self.class_weights = self.class_weights.to(input.device)

        if self.num_labels == 1:
            cross_entropy = tF.binary_cross_entropy_with_logits(input, target.float(), pos_weight=self.class_weights)
        elif self.num_labels > 1:
            cross_entropy = tF.cross_entropy(input, target, weight=self.class_weights)
        else:
            raise ValueError(f'Invalid num_labels: {self.num_labels}')

        return cross_entropy


class DiceLoss(nn.Module):
    def __init__(self, num_labels=None):
        super(DiceLoss, self).__init__()
        self.num_labels = num_labels

    def forward(self, input, target):
        if self.num_labels == 1:
            dice = tmF.dice(input, target)
        elif self.num_labels > 1:
            dice = tmF.dice(input, target, num_classes=self.num_labels, average='macro')
        else:
            raise ValueError(f'Invalid num_labels: {self.num_labels}')

        return 1 - dice


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_labels=None, class_weights=None):
        super(DiceCrossEntropyLoss, self).__init__()
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.dice = DiceLoss(num_labels=num_labels)
        self.cross_entropy = CrossEntropyLoss(num_labels=num_labels, class_weights=class_weights)

    def forward(self, input, target):
        dice = self.dice(input, target)
        cross_entropy = self.cross_entropy(input, target)

        return dice + cross_entropy


class JaccardLoss(nn.Module):
    def __init__(self, num_labels=None):
        super(JaccardLoss, self).__init__()
        self.num_labels = num_labels

    def forward(self, input, target):
        if self.num_labels == 1:
            jaccard = tmF.classification.binary_jaccard_index(input, target)
        elif self.num_labels > 1:
            jaccard = tmF.classification.multiclass_jaccard_index(input, target, num_classes=self.num_labels)
        else:
            raise ValueError(f'Invalid num_labels: {self.num_labels}')

        return 1 - jaccard


class JaccardCrossEntropyLoss(nn.Module):
    def __init__(self, num_labels=None, class_weights=None):
        super(JaccardCrossEntropyLoss, self).__init__()
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.jaccard = JaccardLoss(num_labels=num_labels)
        self.cross_entropy = CrossEntropyLoss(num_labels=num_labels, class_weights=class_weights)

    def forward(self, input, target):
        jaccard = self.jaccard(input, target)
        cross_entropy = self.cross_entropy(input, target)

        return jaccard + cross_entropy


class JaccardBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(JaccardBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, labels, smooth=1):
        logits = logits.view(-1)
        outputs = tF.sigmoid(logits)
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        intersection = (outputs * labels).sum()
        total = (outputs + labels).sum()
        jaccard = (intersection + smooth) / (total - intersection + smooth)

        pos_weight = self.pos_weight.to(logits.device)
        bce = tF.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        return jaccard + bce
