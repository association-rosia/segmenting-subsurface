from torchmetrics.classification import Dice
from torchmetrics.functional.classification import dice
from torch import Tensor


class SegSubDice(Dice):
    def update(self, preds: Tensor, target: Tensor) -> None:
        
        super().update(preds, target)