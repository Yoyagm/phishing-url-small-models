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

PhiUSIIL Phishing URL Dataset by A. Prasad and S. Chandra
([UCI id 967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), reduced to 219,036 unique canonical hosts
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
On the out-of-distribution + temporal test (131,396 hosts), the best small neural model, **DistilBERT (fine-tuned)**, reaches **71.1 % recall at a 1 % false-positive rate** (95 % CI [62.2, 78.7], mean of 3 seeds), against **70.5 %** for the published LightGBM: a paired difference of +0.6 points [-6.7, +6.1]. Against LightGBM trained on the same rows and averaged over the same seeds, the difference is -2.1 points [-8.3, +2.7].
<!-- END:headline -->

<!-- BEGIN:results -->
| Model | Seeds | Test set | ROC-AUC [95 % CI] | Recall @ FPR 1 % [95 % CI] | Recall per seed | TP | FP | Recall @ FPR 0.1 % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM, published (reproduced) | 1 (42) | OOD + temporal | 0.9372 [0.912, 0.954] | **70.5 %** [61.2, 76.6] | — | 22,149 | 999 | 50.8 % |
| LightGBM, published rows, seeds 0-2 | 3 (0, 1, 2) | OOD + temporal | 0.9374 [0.912, 0.954] | **72.0 %** [62.1, 78.4] | 72.8 / 70.7 / 72.5 | 22,610 | 1,000 | 51.4 % |
| LightGBM, neural models' rows | 3 (0, 1, 2) | OOD + temporal | 0.9378 [0.912, 0.954] | **73.2 %** [62.9, 79.9] | 72.5 / 73.2 / 73.9 | 22,982 | 988 | 51.8 % |
| char-CNN (from scratch) | 3 (0, 1, 2) | OOD + temporal | 0.9360 [0.912, 0.952] | **59.9 %** [49.3, 69.1] | 57.2 / 60.7 / 61.8 | 18,800 | 1,000 | 22.4 % |
| BiGRU (from scratch) | 3 (0, 1, 2) | OOD + temporal | 0.9400 [0.917, 0.955] | **68.3 %** [58.9, 75.9] | 70.0 / 66.7 / 68.2 | 21,451 | 1,000 | 24.0 % |
| DistilBERT (fine-tuned) | 3 (0, 1, 2) | OOD + temporal | 0.9488 [0.929, 0.962] | **71.1 %** [62.2, 78.7] | 68.5 / 70.1 / 74.7 | 22,324 | 1,000 | 44.5 % |
| LightGBM, published (reproduced) | 1 (42) | Internal hold-out | 0.9187 [0.856, 0.949] | **72.6 %** [51.2, 83.0] | — | 17,004 | 269 | 63.5 % |
| LightGBM, published rows, seeds 0-2 | 3 (0, 1, 2) | Internal hold-out | 0.9186 [0.856, 0.949] | **72.6 %** [51.1, 83.0] | 72.5 / 72.6 / 72.6 | 17,001 | 269 | 62.9 % |
| LightGBM, neural models' rows | 3 (0, 1, 2) | Internal hold-out | 0.9185 [0.856, 0.949] | **72.5 %** [51.0, 82.9] | 72.4 / 72.5 / 72.6 | 16,985 | 268 | 63.2 % |
| char-CNN (from scratch) | 3 (0, 1, 2) | Internal hold-out | 0.9420 [0.897, 0.964] | **77.3 %** [59.7, 85.7] | 77.3 / 77.2 / 77.3 | 18,100 | 268 | 69.2 % |
| BiGRU (from scratch) | 3 (0, 1, 2) | Internal hold-out | 0.9391 [0.892, 0.962] | **77.0 %** [59.1, 85.6] | 76.9 / 77.1 / 76.9 | 18,027 | 269 | 68.8 % |
| DistilBERT (fine-tuned) | 3 (0, 1, 2) | Internal hold-out | 0.9501 [0.912, 0.969] | **78.5 %** [61.8, 86.7] | 78.8 / 78.4 / 78.4 | 18,395 | 269 | 72.3 % |
<!-- END:results -->

## Risks and limitations

### False positives are the expensive error, and the threshold does not travel

A false positive blocks or warns on a legitimate site. The 1 % operating point in the table is read
off each test set's own ROC curve. When the threshold is instead fixed on held-out validation data
— as any deployment must — and carried to the out-of-distribution set, the false-positive rate it
produces is several times the target:

<!-- BEGIN:threshold -->
| Model | OOD recall, threshold read off the OOD curve | OOD recall, threshold fixed on validation | OOD FPR actually incurred | False positives per 100,000 benign |
|---|---:|---:|---:|---:|
| LightGBM, neural models' rows | 73.2 % | 85.7 % | **6.56 %** | 6,560 |
| char-CNN (from scratch) | 59.9 % | 87.7 % | **11.96 %** | 11,964 |
| BiGRU (from scratch) | 68.3 % | 87.5 % | **10.12 %** | 10,122 |
| DistilBERT (fine-tuned) | 71.1 % | 89.6 % | **12.43 %** | 12,426 |
<!-- END:threshold -->

Re-calibrate on the traffic the model will actually see, and monitor the false-positive rate after
deployment.

### Who gets flagged

The benign hosts the models flag are not a random sample of the web: they are disproportionately
hosts with digits, hyphens or internationalised (punycode) labels:

<!-- BEGIN:false_positives -->
| Benign hosts | Count | Contain a digit | Punycode (`xn--`) | Contain a hyphen | Mean length |
|---|---:|---:|---:|---:|---:|
| All benign OOD hosts | 100,000 | 9.6 % | 0.1 % | 8.4 % | 12.7 |
| Flagged by LightGBM, published (reproduced) | 999 | 64.5 % | 10.7 % | 25.3 % | 15.5 |
| Flagged by LightGBM, published rows, seeds 0-2 | 1,000 | 58.8 % | 11.3 % | 20.5 % | 15.4 |
| Flagged by LightGBM, neural models' rows | 988 | 61.0 % | 10.3 % | 20.1 % | 15.3 |
| Flagged by char-CNN (from scratch) | 1,000 | 85.2 % | 10.7 % | 31.7 % | 15.7 |
| Flagged by BiGRU (from scratch) | 1,000 | 82.1 % | 10.4 % | 32.0 % | 16.2 |
| Flagged by DistilBERT (fine-tuned) | 1,000 | 71.9 % | 12.1 % | 29.2 % | 16.1 |
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
