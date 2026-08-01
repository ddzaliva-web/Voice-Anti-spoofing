import torch
from torch import nn
import torch.nn.functional as F


class LCNNLoss(nn.Module):
    """
    LCNN loss function
    """

    def __init__(self, margin_bona=0.2, margin_spoof=0.5, scale=30.0):
        super().__init__()
        self.margin_bona = margin_bona
        self.margin_spoof = margin_spoof
        self.scale = scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        logits_m = logits.clone()
        mask_bona = (labels == 0)
        mask_spoof = (labels == 1)
        logits_m[mask_bona, 0] -= self.margin_bona
        logits_m[mask_spoof, 1] -= self.margin_spoof
        scores = self.scale * logits_m
        loss = F.cross_entropy(scores, labels)
        return {"loss": loss}