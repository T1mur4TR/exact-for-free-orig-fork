import torch
from .common import logits_deltas

class EXAQOOT(torch.nn.Module):
    def __init__(self, sample_size=256, batch_norm=True, sigma=1., margin=None, reduction='mean'):
        super().__init__()
        self._batch_norm = batch_norm
        self._sigma = sigma
        self._margin = margin
        self._reduction = reduction
        self._shifts = torch.erfinv(torch.linspace(-1., 1., sample_size + 2)[1:-1])

    def forward(self, logits, labels, sigma=None):
        if sigma is None:
            sigma = self._sigma
        if self._shifts.device != logits.device:
            self._shifts = self._shifts.to(logits.device)
        invsqrt2 = 0.70710678118654757273731092936941422522068023681640625 # 2^(-1/2)
        if self._batch_norm:
            logits = torch.nn.functional.batch_norm(logits, None, None, training=True)
        deltas = logits_deltas(logits, labels)
        if self._margin is not None:
            deltas = torch.clip(deltas, max=self._margin)
        loss = -torch.mean(torch.prod(.5 + .5 * torch.erf(self._shifts + deltas.unsqueeze(-1) / sigma * invsqrt2), dim=-2), dim=-1)
        if self._reduction == 'mean':
            loss = loss.mean()
        elif self._reduction == 'sum':
            loss = loss.sum()
        return loss #scale_grad(loss, sigma**2) # <- gradient normalization