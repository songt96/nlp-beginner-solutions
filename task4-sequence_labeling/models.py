import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMTagger(nn.Module):

    def __init__(self, config, embedding=None):
        super(BiLSTMTagger, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=0, _weight=embedding)

        # 2. LSTM Layer
        self.encoder = nn.LSTM(config.embedding_dim, config.hidden_size,
                               bidirectional=True, num_layers=1, batch_first=False)

        # 3. Optional dropout layer
        self.dropout = nn.Dropout(p=config.dropout)

        # 4. Dense Layer
        self.hidden2label = nn.Linear(2*config.hidden_size, config.num_labels)

    def forward(self, *inputs):
        seq = inputs[0]
        embedded = self.embedding(seq)

        encoded, _ = self.encoder(embedded)
        encoded = self.dropout(encoded)

        logits = self.hidden2label(encoded)
        return logits


def token_clf_ce_with_logits(logits, labels, mask):
    """
        logtis: (batch_size, seq_len, num_labels)
        labels: (batch_size, seq_len)
        mask: (batch_size * seq_len)
    """
    num_labels = logits.size(2)
    active_logits = logits.view(-1, num_labels)[mask]
    active_labels = labels.view(-1)[mask]
    loss = F.cross_entropy(active_logits, active_labels)
    return loss
