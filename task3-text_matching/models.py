import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)


def replace_masked_values(tensor: torch.Tensor, mask: torch.ByteTensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.
    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).byte(), replace_with)``.
    """
    if tensor.dim() != mask.dim():
        raise ("tensor.dim() (%d) != mask.dim() (%d)" %
               (tensor.dim(), mask.dim()))
    return tensor.masked_fill(mask, replace_with)


def soft_align_attention(a, b, mask_a=None, mask_b=None):
    mask = mask_a is not None and mask_b is not None
    # (batch_size, len1, len2)
    matrix = torch.matmul(a, b.transpose(1, 2))
    if mask:  # mask before normalization
        # (batch_size, len1, len2)
        a2b_weight = F.softmax(replace_masked_values(matrix, mask_b.eq(0).unsqueeze(1), float('-inf')),
                               dim=2)
        # (batch_size, len2, len1)
        b2a_weight = F.softmax(replace_masked_values(matrix, mask_a.eq(0).unsqueeze(2), float('-inf')),
                               dim=1).transpose(1, 2)
    else:
        # (batch_size, len1, len2)
        a2b_weight = F.softmax(matrix, dim=2)
        # (batch_size, len2, len1)
        b2a_weight = F.softmax(matrix, dim=1).transpose(1, 2)
    # (batch_size, len1, dim)
    aligned_a = torch.matmul(a2b_weight, b)
    # (batch_size, len2, dim)
    aligned_b = torch.matmul(b2a_weight, a)
    return aligned_a, aligned_b


class Esim(nn.Module):
    def __init__(self, config, embedding=None):
        super(Esim, self).__init__()
        self.static_emb = config.dict.get('static_emb', True)
        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=0, _weight=embedding)
        if self.static_emb is True:
            self.embedding.weight.requires_grad = False
        logger.info('Embedding\'s weight requires gradient: {}'.format(
            self.embedding.weight.requires_grad))
        self.encoder = nn.LSTM(
            config.embedding_dim, config.hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size * 8, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.inference = nn.LSTM(
            config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.inference_dropout = nn.Dropout(config.dropout)
        self.hidden = nn.Sequential(
            nn.Linear(config.hidden_size * 8, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, ids_a, ids_b):
        mask_a, mask_b = ids_a.ne(0), ids_b.ne(0)
        embedded_a = self.embedding(ids_a)
        embedded_b = self.embedding(ids_b)

        encoded_a, _ = self.encoder(embedded_a)
        encoded_b, _ = self.encoder(embedded_b)

        aligned_a, aligned_b = soft_align_attention(
            encoded_a, encoded_b, mask_a, mask_b)

        enhanced_a = torch.cat(
            [encoded_a, aligned_a, encoded_a - aligned_a, encoded_a * aligned_a], dim=2)
        enhanced_b = torch.cat(
            [encoded_b, aligned_b, encoded_b - aligned_b, encoded_b * aligned_b], dim=2)

        projected_a = self.projection(enhanced_a)
        projected_b = self.projection(enhanced_b)

        inference_a, _ = self.inference(projected_a)
        inference_b, _ = self.inference(projected_b)
        inference_a = self.inference_dropout(inference_a)
        inference_b = self.inference_dropout(inference_b)

        avg_a = torch.sum(inference_a * mask_a.unsqueeze(2).float(),
                          dim=1) / mask_a.float().sum()
        avg_b = torch.sum(inference_b * mask_b.unsqueeze(2).float(),
                          dim=1) / mask_b.float().sum()

        max_a, _ = torch.max(replace_masked_values(
            inference_a, mask_a.eq(0).unsqueeze(2), float('-inf')), dim=1)
        max_b, _ = torch.max(replace_masked_values(
            inference_b, mask_b.eq(0).unsqueeze(2), float('-inf')), dim=1)
        v = torch.cat([avg_a, max_a, avg_b, max_b], dim=1)
        hidden = self.hidden(v)

        logits = self.classifier(hidden)
        return logits
