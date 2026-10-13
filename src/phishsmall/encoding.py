"""Turn host names into padded integer tensors, the one input format every neural model takes.

Both the character models and the transformer consume ``(ids, lengths)``: a
``[n, max_len]`` matrix padded with the pad id, and each row's true length.
Encoding the whole set once, up front, keeps tokenisation out of the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Fixed, not fitted: a vocabulary learned from the data would be one more place
# for the training split to shape the encoding. The canonical hosts in every set
# here use only these characters (measured: 1 non-alphabet character in 6.2M).
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789.-_"
PAD_ID = 0
UNKNOWN_ID = 1
CHAR_TO_ID = {character: index + 2 for index, character in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 2

# The longest canonical host in any set is 110 characters.
MAX_CHARS = 128


@dataclass(frozen=True)
class Encoded:
    ids: torch.Tensor
    lengths: torch.Tensor

    def __len__(self) -> int:
        return len(self.lengths)

    def batch(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows ``index``, trimmed to the longest of them so no step pays for global padding."""
        lengths = self.lengths[index]
        width = int(lengths.max())
        return self.ids[index, :width], lengths


def clip_host(host: str, limit: int = MAX_CHARS) -> str:
    """Keep both ends of an over-long host.

    The left end carries the impersonation (``paypal.com.``…) and the right end the
    registrable domain and suffix; the middle is the least informative part to lose.
    """
    if len(host) <= limit:
        return host
    head = limit // 2
    return host[:head] + host[len(host) - (limit - head) :]


def encode_chars(hosts: list[str], limit: int = MAX_CHARS) -> Encoded:
    ids = torch.full((len(hosts), limit), PAD_ID, dtype=torch.long)
    lengths = torch.ones(len(hosts), dtype=torch.long)
    for row, host in enumerate(hosts):
        clipped = clip_host(host, limit)
        if not clipped:
            # An empty host still needs one position, or pooling over it is undefined.
            ids[row, 0] = UNKNOWN_ID
            continue
        ids[row, : len(clipped)] = torch.tensor(
            [CHAR_TO_ID.get(character, UNKNOWN_ID) for character in clipped]
        )
        lengths[row] = len(clipped)
    return Encoded(ids=ids, lengths=lengths)


def encode_tokens(hosts: list[str], tokenizer, limit: int) -> Encoded:
    """Subword-encode hosts with a Hugging Face tokenizer, special tokens included."""
    encoded = tokenizer(hosts, truncation=True, max_length=limit, padding=False)["input_ids"]
    ids = torch.full((len(hosts), limit), tokenizer.pad_token_id, dtype=torch.long)
    lengths = torch.empty(len(hosts), dtype=torch.long)
    for row, sequence in enumerate(encoded):
        ids[row, : len(sequence)] = torch.tensor(sequence)
        lengths[row] = len(sequence)
    return Encoded(ids=ids, lengths=lengths)
