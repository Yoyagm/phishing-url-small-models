"""The neural models compared against LightGBM.

All three share one interface — ``forward(ids, lengths) -> logits`` of shape
``[batch]`` — so a single training loop and a single scorer serve them all.
Scores are kept as logits, never squashed through a sigmoid: in float32 a sigmoid
saturates to exactly 1.0 for confident phishing, and ties at the top of the
ranking are precisely where recall at a 1 % false-positive rate is decided.
"""

from __future__ import annotations

import torch
from torch import nn

from phishsmall.encoding import PAD_ID, VOCAB_SIZE


def _valid_positions(lengths: torch.Tensor, width: int) -> torch.Tensor:
    """``[batch, width]`` mask, True where a position holds a real character."""
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]


class CharCNN(nn.Module):
    """Character embeddings, parallel convolutions of widths 2-5, global max pool.

    The widths cover the n-grams that matter in host names — ``xn--``, ``-login``,
    digit runs — without the model ever being told which ones those are.
    """

    def __init__(self, embedding_dim: int = 32, filters: int = 128,
                 widths: tuple[int, ...] = (2, 3, 4, 5), hidden: int = 128,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_ID)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters, width, padding=width - 1) for width in widths
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(filters * len(widths), hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(ids).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            activated = torch.relu(convolution(embedded))
            # 'Full' padding gives every host windows that overhang both of its ends
            # by the same amount. Windows touching no real character are masked, so a
            # host scores the same whichever longer hosts share its batch.
            width = convolution.kernel_size[0]
            mask = _valid_positions(lengths + width - 1, activated.shape[2])
            pooled.append(activated.masked_fill(~mask[:, None, :], float("-inf")).amax(dim=2))
        return self.head(torch.cat(pooled, dim=1)).squeeze(1)


class BiGRU(nn.Module):
    """Bidirectional GRU over characters, with masked max and mean pooling."""

    def __init__(self, embedding_dim: int = 32, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_ID)
        self.gru = nn.GRU(embedding_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(ids)
        # Packing makes the backward direction start at each host's real last
        # character instead of reading through the padding first.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=ids.shape[1]
        )
        mask = _valid_positions(lengths, ids.shape[1])[:, :, None]
        maximum = output.masked_fill(~mask, float("-inf")).amax(dim=1)
        mean = (output * mask).sum(dim=1) / lengths[:, None].to(output.dtype)
        return self.head(torch.cat([maximum, mean], dim=1)).squeeze(1)


class PretrainedClassifier(nn.Module):
    """A Hugging Face encoder with a one-logit classification head, fully fine-tuned."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        attention_mask = _valid_positions(lengths, ids.shape[1]).long()
        return self.backbone(input_ids=ids, attention_mask=attention_mask).logits.squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
