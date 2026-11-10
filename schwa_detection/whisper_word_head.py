"""A Schwa head with trainable embeddings of audio-inferred words."""
import torch
from torch import nn

from improve import PhoneHead


class WordAwareHead(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim=32, head_config=None):
        super().__init__()
        if vocabulary_size < 1 or vocabulary_size >= 2 ** 24:
            raise ValueError('Word IDs must fit exactly in float32.')
        config = dict(head_config or {})
        if config.get('input_dim') != 1424 + embedding_dim:
            raise ValueError('Expected original 1424 features plus learned word embedding.')
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.head = PhoneHead(**config)

    def forward(self, values):
        if values.ndim != 3 or values.shape[-1] != 1425:
            raise ValueError('Expected 1424 acoustic/lexical features and one integer word ID.')
        raw_ids = values[..., -1]
        if (not torch.isfinite(raw_ids).all() or torch.any(raw_ids != raw_ids.round())
                or torch.any(raw_ids < 0) or torch.any(raw_ids >= self.vocabulary_size)):
            raise ValueError('Word IDs must be finite in-vocabulary integers.')
        embedded = self.embedding(raw_ids.long())
        return self.head(torch.cat([values[..., :1424], embedded], dim=-1))
