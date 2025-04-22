import torch
import math

class DOUCE(torch.nn.Module):
    def __init__(self, batch_norm=True, delta=0., sigma=1., reduction='mean', label_smoothing=0., soft_labels_correction=None):
        super().__init__()
        self._batch_norm = batch_norm
        self._delta = delta
        self._sigma = sigma
        self._reduction = reduction
        self._label_smoothing = label_smoothing
        self._soft_labels_correction = soft_labels_correction if soft_labels_correction is not None else label_smoothing > 0.

    def forward(self, logits, labels, delta=None, sigma=None):
        if delta is None:
            delta = self._delta
        if sigma is None:
            sigma = self._sigma
        if self._batch_norm:
            logits = torch.nn.functional.batch_norm(logits, None, None, training=True)
        if delta == 0.: # more numerically stable
            return torch.nn.functional.cross_entropy(logits / sigma, labels, reduction=self._reduction, label_smoothing=self._label_smoothing)
        logprobs = -torch.log(torch.softmax(logits / sigma, dim=-1) + delta)
        if logits.shape == labels.shape:
            loss = (logprobs * labels).sum(dim=-1)
        else:
            loss = logprobs.take_along_dim(labels.unsqueeze(-1), -1).squeeze()
        if self._label_smoothing != 0.:
            loss = (1. - self._label_smoothing) * loss + self._label_smoothing * logprobs.mean(dim=-1)
        if self._soft_labels_correction:
            loss = loss + delta * torch.sum(logprobs, dim=-1)
        if self._reduction == 'mean':
            loss = loss.mean()
        elif self._reduction == 'sum':
            loss = loss.sum()
        return (delta + 1.) * (math.log(delta + 1.) + loss)
    