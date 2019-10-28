import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmLM(nn.Module):
    def __init__(self, config, embedding=None):
        super(LstmLM, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding = nn.Sequential(
            nn.Embedding(self.vocab_size, config.embedding_dim,
                         padding_idx=0, _weight=embedding),
            nn.Dropout(config.dropout)
        )
        self.encoder = nn.LSTM(
            config.embedding_dim, config.hidden_size, config.num_layers, batch_first=True)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # self.decoder.weight = self.embedding[0].weight

    def forward(self, ids, hidden=None):
        # （batch_size, seq_len, embedding_dim)
        embedded = self.embedding(ids)
        # encoded （batch_size, seq_len, hidden_size)
        if hidden is not None:
            encoded, hidden = self.encoder(embedded, hidden)
        else:
            encoded, hidden = self.encoder(embedded)
        # decoded （batch_size, seq_len, vocab_size) -> (batch_size*seq_len, vocab_size)
        decoded = self.decoder(encoded)
        return decoded, hidden
