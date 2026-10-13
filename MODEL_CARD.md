# Model card — host-name phishing classifiers (char-CNN, BiGRU, DistilBERT, LightGBM)

This card covers every model trained in this repository. They are **research artefacts for a
comparison**, not a product. No weights are published: they are reproducible with `make train`,
and the DistilBERT checkpoints alone are ~250 MB each.

## Intended use

- **Intended:** studying how model families compare on host-name phishing detection under a
  leak-free evaluation; a reproducible baseline for follow-up work on the same splits.
- **Acceptable with care:** one signal among several in a triage pipeline, behind a human or a
  second system, with the threshold re-calibrated on the deployment's own traffic (see *Risks*).
- **Not intended:** blocking traffic on its own; judging a full URL, page or e-mail; any setting
  where a "legitimate" verdict is read as a safety guarantee.

## Inputs and outputs

- **Input:** one canonical host name (lower-cased; scheme, user-info, port, path, query, fragment,
  trailing dot and a leading `www.` removed) — the canonicalisation of phishing-url-detector,
  vendored unchanged. The models never see anything else.
- **Output:** a score; higher means more likely phishing. Neural models return a logit, LightGBM a
  probability. Neither is calibrated as a probability for any deployment.

## Models

| Model | Parameters | Input | Training |
|---|---:|---|---|
| char-CNN | ~125 k | characters (fixed 39-symbol alphabet) | from scratch, AdamW, ≤ 20 epochs, early stopping |
| BiGRU | ~55 k | characters | from scratch, AdamW, ≤ 20 epochs, early stopping |
| DistilBERT | ~67 M | WordPiece subwords (≤ 64 tokens) | `distilbert-base-uncased` at a pinned revision, full fine-tuning, ≤ 3 epochs |
| LightGBM | 300 trees | 51 hand-built host features | parameters of phishing-url-detector, unchanged |

Early stopping uses ROC-AUC on a validation split grouped by registrable domain and disjoint from
both test sets. Three seeds per neural model. Exact settings live in `src/phishsmall/neural.py` and
`reports/results.json`.

## Training data

PhiUSIIL Phishing URL Dataset (UCI id 967, CC BY 4.0), reduced to 219,036 unique canonical hosts
after dropping 58 hosts carrying both labels. The neural models train on 153,180 hosts; the
remaining training-portion domains form the validation split. PhiUSIIL's legitimate class is a
collection artefact at the URL level (every legitimate row is `https://www.<domain>` with no
path) — the reason only host names are used.

## Evaluation data

- **Internal hold-out:** 50,332 PhiUSIIL hosts whose registrable domains appear nowhere in training.
- **Out-of-distribution + temporal:** 100,000 Tranco top-1M hosts absent from PhiUSIIL, and 31,396
  PhishTank hosts submitted after 2024-03-04 (PhiUSIIL's collection cutoff); hosts on both lists
  removed.
- **OpenPhish:** the live community feed on 2026-08-14, phishing only.

## Metrics

Primary: **recall at a 1 % false-positive rate**. Also ROC-AUC, recall at 0.1 % FPR, and absolute
true/false positives at the operating point. Intervals are 95 % bootstrap intervals over
registrable domains (1,000 resamples), and the mean is over seeds.

<!-- BEGIN:headline -->

<!-- END:headline -->

<!-- BEGIN:results -->

<!-- END:results -->

## Risks and limitations

### False positives are the expensive error, and the threshold does not travel

A false positive blocks or warns on a legitimate site. The 1 % operating point in the table is read
off each test set's own ROC curve. When the threshold is instead fixed on held-out validation data
— as any deployment must — and carried to the out-of-distribution set, the false-positive rate it
produces is several times the target:

<!-- BEGIN:threshold -->

<!-- END:threshold -->

Re-calibrate on the traffic the model will actually see, and monitor the false-positive rate after
deployment.

### Who gets flagged

The benign hosts the models flag are not a random sample of the web: they are disproportionately
hosts with digits, hyphens or internationalised (punycode) labels:

<!-- BEGIN:false_positives -->

<!-- END:false_positives -->

Sites whose names look machine-generated, and sites in non-Latin scripts, carry more of the
false-positive cost. Anyone deploying such a model should measure this on their own population.

### Other limits

- **Host names only:** phishing on compromised legitimate domains, behind URL shorteners, or on
  shared hosting with an innocuous subdomain is invisible to these models.
- **One training source and one capture date.** Phishing infrastructure shifts; scores degrade as
  the feeds move on from 2026-08-14.
- **Not adversarially tested.** An attacker who knows the model can choose host names it scores low.
- **Three seeds; MPS non-determinism.** Neural results move between seeds and, slightly, between
  re-runs on Apple-silicon GPUs.

## Ethical and legal notes

The training and evaluation data are public phishing and popularity lists. No personal data is
used. The third-party data is not redistributed here; each source's terms apply (see the README).
