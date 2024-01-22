from typing import Callable, Optional, Tuple, Union, Any, List

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


def poly_loss(
        x: Tensor,
        target: Tensor,
        eps: float = 2.0,
        class_weights: Optional[Tensor] = None,
        instance_weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
) -> Tensor:
    """Implements the Poly1 loss from `"PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions" <https://arxiv.org/pdf/2204.12511.pdf>`_.
    Args:
        x (torch.Tensor[N, K, ...]): predicted probability
        target (torch.Tensor[N, K, ...]): target probability
        eps (float, optional): epsilon 1 from the paper
        class_weights (torch.Tensor[K], optional): manual rescaling of each class
        weight_instance (torch.Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method
    Returns:
        torch.Tensor: loss reduced with `reduction` method
    """

    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=1)
    # logpt = F.softmax(x, dim=1)/0.1

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
    # Ignore index (set loss contribution to 0)
    if instance_weight is not None:
        valid_idxs = instance_weight > 0
    else:
        valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)
        if ignore_index >= 0 and ignore_index < x.shape[1]:
            valid_idxs[target.view(-1) == ignore_index] = False

    # Get P(class)
    loss = -1 * logpt + eps * (1 - logpt.exp())

    # Weight
    if class_weights is not None:
        # Tensor type
        if class_weights.type() != x.data.type():
            class_weights = class_weights.type_as(x.data)
        logpt = class_weights.gather(0, target.data.view(-1)) * logpt

    if instance_weight is not None:
        logpt = instance_weight * logpt

    # Loss reduction
    if reduction == "sum":
        loss = loss[valid_idxs].sum()
    elif reduction == "mean":
        loss = loss[valid_idxs].mean()
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)

    return loss


class _Loss(nn.Module):
    def __init__(
            self,
            class_weights: Optional[Union[float, List[float], Tensor]] = None,
            ignore_index: int = -100,
            reduction: str = "mean",
    ) -> None:
        super().__init__()
        # Cast class weights if possible
        self.weight: Optional[Tensor]
        if isinstance(class_weights, (float, int)):
            self.register_buffer("weight", torch.Tensor([class_weights, 1 - class_weights]))
        elif isinstance(class_weights, list):
            self.register_buffer("weight", torch.Tensor(class_weights))
        elif isinstance(class_weights, Tensor):
            self.register_buffer("weight", class_weights)
        else:
            self.weight = None
        self.ignore_index = ignore_index
        # Set the reduction method
        if reduction not in ["none", "mean", "sum"]:
            raise NotImplementedError("argument reduction received an incorrect input")
        self.reduction = reduction


class PolyLoss(_Loss):
    """Implements the Poly1 loss from `"PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions" <https://arxiv.org/pdf/2204.12511.pdf>`_.
    Args:
        weight (torch.Tensor[K], optional): class weight for loss computation
        eps (float, optional): epsilon 1 from the paper
        ignore_index: int = -100,
        reduction: str = 'mean',
    """

    def __init__(
            self,
            *args: Any,
            eps: float = 2.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, x: Tensor, target: Tensor, instance_weight: Tensor = None, reduction=None,
                class_weights=None) -> Tensor:
        self.reduction = reduction if reduction is not None else self.reduction
        self.weight = class_weights if class_weights is not None else self.weight
        return poly_loss(x, target, self.eps, self.weight, instance_weight, self.ignore_index, self.reduction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps}, reduction='{self.reduction}')"
