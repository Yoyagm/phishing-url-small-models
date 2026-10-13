"""Neural inputs, models and the training loop — on CPU, with nothing downloaded.

The transformer path is exercised with a 2-layer BERT built from a config and a
tokenizer built from a 60-entry vocabulary written to a temp directory, so the
suite needs no network and no pretrained weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phishsmall.encoding import (
    CHAR_TO_ID,
    PAD_ID,
    UNKNOWN_ID,
    clip_host,
    encode_chars,
    encode_tokens,
)
from phishsmall.models import BiGRU, CharCNN, PretrainedClassifier
from phishsmall.training import FitConfig, fit, length_pooled_batches, predict

CPU = torch.device("cpu")
HOSTS = ["a.co", "paypal.com.secure-login.xyz", "x", "", "bücher.de"]


def test_character_encoding_pads_and_marks_unknowns() -> None:
    encoded = encode_chars(HOSTS)
    assert encoded.lengths.tolist() == [4, 27, 1, 1, 9]
    assert encoded.ids[0, :4].tolist() == [CHAR_TO_ID[c] for c in "a.co"]
    assert encoded.ids[0, 4:].eq(PAD_ID).all()
    assert encoded.ids[3, 0] == UNKNOWN_ID  # empty host keeps one position
    assert encoded.ids[4, 1] == UNKNOWN_ID  # 'ü' is outside the fixed alphabet


def test_over_long_hosts_keep_both_ends() -> None:
    host = "paypal." + "a" * 300 + ".evil.tk"
    clipped = clip_host(host, limit=40)
    assert len(clipped) == 40
    assert clipped.startswith("paypal.")
    assert clipped.endswith(".evil.tk")


def test_batches_are_trimmed_to_their_longest_member() -> None:
    encoded = encode_chars(HOSTS)
    ids, lengths = encoded.batch(torch.tensor([0, 2]))
    assert ids.shape == (2, 4)
    assert lengths.tolist() == [4, 1]


@pytest.mark.parametrize("architecture", [CharCNN, BiGRU])
def test_a_host_scores_the_same_whatever_shares_its_batch(architecture) -> None:
    """Padding must be invisible: a short host alone and next to a long one, same logit."""
    torch.manual_seed(0)
    model = architecture().eval()
    encoded = encode_chars(HOSTS)
    with torch.no_grad():
        alone = model(*encoded.batch(torch.tensor([0])))
        together = model(*encoded.batch(torch.tensor([0, 1])))
    assert together.shape == (2,)
    assert torch.allclose(alone[0], together[0], atol=1e-6)


def test_length_pooled_batches_cover_every_row_once_and_repeat_per_seed() -> None:
    lengths = torch.randint(1, 60, (1_000,), generator=torch.Generator().manual_seed(0))
    first = length_pooled_batches(lengths, 32, torch.Generator().manual_seed(5), pool_batches=4)
    again = length_pooled_batches(lengths, 32, torch.Generator().manual_seed(5), pool_batches=4)
    assert sorted(torch.cat(first).tolist()) == list(range(1_000))
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))


def test_predict_returns_scores_in_input_order() -> None:
    torch.manual_seed(0)
    model = CharCNN().eval()
    encoded = encode_chars(HOSTS)
    batched = predict(model, encoded, CPU, batch_size=2)
    one_by_one = [predict(model, encode_chars([host]), CPU, batch_size=1)[0] for host in HOSTS]
    assert batched.dtype == np.float64
    assert np.allclose(batched, one_by_one, atol=1e-6)


def _toy_problem(n: int = 256):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, n)
    hosts = [f"login-verify{i}.secure.xyz" if y else f"site{i}.com" for i, y in enumerate(labels)]
    return hosts, labels


def test_training_learns_a_toy_problem_and_restores_the_best_epoch() -> None:
    torch.manual_seed(0)
    hosts, labels = _toy_problem()
    config = FitConfig(max_epochs=4, patience=2, batch_size=32, learning_rate=5e-3, eval_batch_size=64)
    model, history, best_epoch = fit(
        CharCNN(), encode_chars(hosts[:192]), labels[:192], encode_chars(hosts[192:]), labels[192:],
        config, CPU, seed=0, log=lambda _: None,
    )
    assert 1 <= best_epoch <= len(history) <= 4
    best = max(entry["val_roc_auc"] for entry in history)
    assert history[best_epoch - 1]["val_roc_auc"] == best
    assert best > 0.95
    scores = predict(model, encode_chars(hosts[192:]), CPU, 64)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(labels[192:], scores) == pytest.approx(best)


@pytest.fixture
def tiny_tokenizer(tmp_path):
    from transformers import BertTokenizerFast

    characters = "abcdefghijklmnopqrstuvwxyz0123456789"
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "-", *characters]
    vocabulary += ["##" + character for character in characters]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocabulary) + "\n")
    return BertTokenizerFast(vocab_file=str(path))


def test_subword_encoding_and_a_tiny_transformer_train_end_to_end(tiny_tokenizer) -> None:
    from transformers import BertConfig, BertForSequenceClassification

    hosts, labels = _toy_problem(128)
    encoded = encode_tokens(hosts, tiny_tokenizer, limit=48)
    assert encoded.ids[:, 0].eq(tiny_tokenizer.cls_token_id).all()
    assert int(encoded.lengths.max()) <= 48

    torch.manual_seed(0)
    config = BertConfig(vocab_size=tiny_tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, num_labels=1)
    model = PretrainedClassifier(BertForSequenceClassification(config))
    fit_config = FitConfig(max_epochs=2, patience=1, batch_size=16, learning_rate=1e-3, eval_batch_size=32)
    _, history, _ = fit(model, encode_tokens(hosts[:96], tiny_tokenizer, 48), labels[:96],
                        encode_tokens(hosts[96:], tiny_tokenizer, 48), labels[96:],
                        fit_config, CPU, seed=0, log=lambda _: None)
    assert len(history) >= 1
    assert all(np.isfinite(entry["train_loss"]) for entry in history)
