import numpy as np
import torch
from functools import partial
import segmentation_models_pytorch as smp
import sys
from torch import Tensor
from pytorch_toolbelt import losses
from torch import nn
from torch.nn.modules.loss import _Loss
from functools import partial


def get_model(name, params):
    if name == 'unet':
        return smp.Unet(**params)
    else:
        raise ValueError("Invalid model name")

def get_loss(name: str, mode: str = None, params: dict = {}):
    """
    Constructs specified by name binary of multiclass loss
    with the defined params
    """
    losses_dict = {
                   'focal': get_focal(mode=mode),
                   'dice': partial(losses.DiceLoss, mode=mode),
                   'bce': nn.BCEWithLogitsLoss,
                   'cross_entropy': losses.SoftCrossEntropyLoss
                   }

    return losses_dict[name](**params)


def get_focal(mode):

    if mode == 'binary':
        return losses.BinaryFocalLoss
    elif mode == "multiclass":
        return losses.FocalLoss
    else:
        raise ValueError("Mode {} is not supported".format(mode))


def get_metric(name: str, params: dict):
        return partial(multiclass_iou_dice_score, metric=name, **params)
    
def binary_iou_dice_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metric='iou',
    apply_sigmoid: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:

    
    assert metric in {'iou', 'dice'}
    assert y_pred.shape == y_true.shape
            
    # Apply sigmoid if needed
    if apply_sigmoid:
        y_pred = torch.sigmoid(y_pred)

    # Make binary predictions
    y_pred = (y_pred > threshold).type(y_true.dtype)
            
    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if metric == 'iou':
        score = (intersection + eps) / (cardinality - intersection + eps)
    else:
        score = (2.0 * intersection + eps) / (cardinality + eps)
        
    return score

def multiclass_iou_dice_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metric='iou',
    threshold: float = 0.35,
    eps=1e-7,
):
    scores = []
    num_classes = y_pred.shape[1]
    y_pred_ = y_pred.log_softmax(dim=1).exp()

    for class_index in range(num_classes):
        y_pred_i = y_pred_[:, class_index, :, :]
        y_true_i = (y_true == class_index)

        score = binary_iou_dice_score(
            y_pred=y_pred_i,
            y_true=y_true_i,
            metric=metric,
            apply_sigmoid=False,
            threshold=threshold,
            eps=eps,
        )
        scores.append(score)

    return np.mean(scores)

def get_scheduler(optimizer: torch.optim, name: str, params: dict, additional_params: dict):
    if name == 'ReduceLROnPlateau':
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params),
            **additional_params
        }
        return scheduler
    if name == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer,  **params)