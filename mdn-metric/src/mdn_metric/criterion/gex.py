import torch

class GEX(torch.nn.Module):
    def __init__(self, batch_norm=True, sigma=1., margin=None, cap=None, reduction='mean', label_smoothing=None, use_k_closest=None):
        super().__init__()
        self._batch_norm = batch_norm
        self._sigma = sigma
        self._margin = margin
        self._cap = cap
        self._reduction = reduction
        self._smoothing_factor = .5 / (1. - label_smoothing) if label_smoothing is not None else None
        self._use_k_closest = use_k_closest

    def forward(self, logits, labels, sigma=None):
        if sigma is None:
            sigma = self._sigma
        if self._batch_norm:
            logits = torch.nn.functional.batch_norm(logits, None, None, training=True)
        if self._use_k_closest is not None:
            if logits.shape == labels.shape:
                raise ValueError("Cannot use soft labels with use_k_closest != None (use_k_closest = {self._use_k_closest}).")
            with torch.no_grad():
                diffs = torch.abs(logits - logits.take_along_dim(labels.unsqueeze(-1), -1))
                diffs[torch.arange(len(labels)), labels] -= 1.
                indices = torch.argsort(diffs, dim=-1)[:, :self._use_k_closest + 1]
                labels = torch.zeros_like(labels)
            logits = logits.take_along_dim(indices, dim=-1)
        if self._margin is not None or self._cap is not None:
            if logits.shape == labels.shape:
                logits = torch.clip(logits - (logits * labels).sum(dim=-1, keepdim=True), min=-self._margin, max=self._cap)
            else:
                logits = torch.clip(logits - logits.take_along_dim(labels.unsqueeze(-1), -1), min=-self._margin, max=self._cap)
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