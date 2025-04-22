import torch

class GEX(torch.nn.Module):
    def __init__(self, batch_norm=True, sigma=1., margin=None, reduction='mean', label_smoothing=None):
        super().__init__()
        self._batch_norm = batch_norm
        self._sigma = sigma
        self._margin = margin
        self._reduction = reduction
        self._smoothing_factor = .5 / (1. - label_smoothing) if label_smoothing is not None else None

    def forward(self, logits, labels, sigma=None):
        if sigma is None:
            sigma = self._sigma
        if self._batch_norm:
            logits = torch.nn.functional.batch_norm(logits, None, None, training=True)
        if self._margin is not None:
            if logits.shape == labels.shape:
                logits = torch.clip(logits - (logits * labels).sum(dim=-1, keepdim=True), min=-self._margin)
            else:
                logits = torch.clip(logits - logits.take_along_dim(labels.unsqueeze(-1), -1), min=-self._margin)
        probs = torch.softmax(logits / sigma, dim=-1)
        if logits.shape == labels.shape:
            loss = -(probs * labels).sum(dim=-1)
        else:
            loss = -probs.take_along_dim(labels.unsqueeze(-1), -1).squeeze()
        if self._smoothing_factor is not None:
            loss = loss + self._smoothing_factor * torch.square(probs).sum(dim=-1)
        if self._reduction == 'mean':
            loss = loss.mean()
        elif self._reduction == 'sum':
            loss = loss.sum()
        return 1. + loss #scale_grad(loss, sigma**2) # <- gradient normalization