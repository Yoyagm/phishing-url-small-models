# Small Neural Models vs LightGBM for Phishing Detection — Evaluated Without Leaks

**Do small neural models beat gradient-boosted trees at detecting phishing from a host name, when
the evaluation cannot be gamed?** This repository trains a character-level CNN and a BiGRU from
scratch, fine-tunes DistilBERT, and compares all three against the LightGBM model of
[phishing-url-detector](https://github.com/Yoyagm/phishing-url-detector) on *that project's own
leak-free evaluation*, reproduced to the last digit. The result is reported as it came out.

<!-- BEGIN:headline -->

<!-- END:headline -->

**Short answer: no.** Fine-tuned DistilBERT ties LightGBM out of distribution, the two models
trained from scratch lose there, and all three win clearly *in* distribution. Why that gap exists,
and what it implies for anyone choosing a model on an in-distribution score, is in
[What won, and why we think so](#what-won-and-why-we-think-so).

---

## The question

phishing-url-detector showed that the most widely used recent phishing-URL benchmark (PhiUSIIL,
UCI id 967) carries two label leaks, and that a host-name LightGBM evaluated without them reaches
**70.5 % recall at a 1 % false-positive rate** on phishing submitted two years after its training
data. That number is the bar here.

A tree model on 51 hand-built features is a strong, cheap baseline. The obvious question for anyone
who works with neural networks is whether a model that reads the raw characters — and so can learn
features nobody wrote down — does better. Published work on phishing-URL classification routinely
reports 99 %+ accuracy for neural models and trees alike, on evaluations of the kind
phishing-url-detector showed to be broken, so it settles nothing.

**The contribution of this repository is the comparison's rigour, not a winner.** Same split, same
test sets, same operating metric, several seeds, domain-level confidence intervals, paired tests,
and a baseline that must reproduce the published figures exactly before any comparison is allowed
to run.

## Method

### Data and splits — reused, not re-implemented

The canonicalisation, Public Suffix List, dataset assembly and LightGBM features are **vendored
byte-for-byte** from phishing-url-detector (`src/phishurl/`, pinned by SHA-256 in
`tests/test_vendored.py`). The splits are drawn with the same `GroupShuffleSplit`, seed and groups:

- **Rows are canonical host names**, never full URLs: the path, scheme and `www.` are where
  PhiUSIIL's collection artefact lives.
- **Grouped by registrable domain** (eTLD+1, ICANN section of the PSL): no domain appears on two
  sides of any split, so a model cannot be rewarded for memorising `*.web.app`.
- **Two test sets.** The *internal hold-out* is the grouped 20 % of PhiUSIIL. The
  *out-of-distribution + temporal* set pairs 100,000 Tranco hosts never seen in PhiUSIIL with
  PhishTank hosts submitted after PhiUSIIL's collection cutoff (2024-03-04); hosts on both lists
  are removed. The OpenPhish live feed is a third, phishing-only probe.
- **A validation split for early stopping** — the only addition — is carved out of the original
  training portion (10 % of its domains, grouped the same way). The test sets are never read
  during training.

<!-- BEGIN:sets -->

<!-- END:sets -->

The last column explains a feature of every result below: the internal hold-out's phishing is
dominated by two hosting platforms, so its confidence intervals are wide.

### Models

| Model | Input | What it is |
|---|---|---|
| LightGBM, published (reproduced) | 51 host features | The original model: same rows, parameters and seed 42. |
| LightGBM, published rows, seeds 0-2 | 51 host features | Same rows and parameters, other seeds: how much of 70.5 % is the seed. |
| LightGBM, neural models' rows | 51 host features | Trained on exactly the rows the neural models see (without the validation domains). |
| char-CNN | characters | Embeddings → parallel convolutions of width 2–5 → global max pool → MLP. Trained from scratch. |
| BiGRU | characters | Embeddings → bidirectional GRU → masked max + mean pool → MLP. Trained from scratch. |
| DistilBERT | WordPiece subwords | `distilbert-base-uncased` (pinned revision), fully fine-tuned with a one-logit head. |

Hyperparameters were fixed before any test result existed and were **not tuned**, on the test sets
or otherwise; they are listed in `src/phishsmall/neural.py`. Every neural model uses AdamW with
warm-up and linear decay, and keeps the weights of its best validation epoch (ROC-AUC on the
grouped validation split). DistilBERT is capped at **3 epochs** to keep each run under ~30 minutes
on an Apple M4; in practice every seed peaked after its first epoch and stopped after the second.
LoRA was not needed: full fine-tuning of 67 M parameters fits comfortably in 16 GB.

<!-- BEGIN:cost -->

<!-- END:cost -->

### Metrics and uncertainty

- **Primary metric: recall at a 1 % false-positive rate**, as in the original. A false positive
  blocks a legitimate site, which is the costly error in deployment. ROC-AUC, recall at 0.1 % FPR
  and the absolute true/false positives at the operating point are reported alongside.
- **Three seeds** per trained model family (0, 1, 2); the table reports the mean and every seed.
- **95 % confidence intervals from 1,000 bootstrap resamples of registrable domains**, not of
  hosts: hosts under one domain are not independent, which is why the split is grouped in the
  first place. Each resample is shared by every model, so **differences are paired**.
- **The reproduction gate.** `make eval` refuses to run unless the reproduced LightGBM matches the
  published ROC-AUC and recall on both test sets to 1e-9:

<!-- BEGIN:reproduction -->

<!-- END:reproduction -->

## Results

<!-- BEGIN:results -->

<!-- END:results -->

TP and FP are counts at the operating point, averaged over seeds. FP sits near 1 % of the benign
hosts for every model by construction; recall is what differs. The intervals above are marginal and
wide — the internal ones especially, because two hosting platforms hold over a third of that set's
phishing — so they overlap even where one model is consistently ahead. The paired differences below
are what decide.

### Paired differences against LightGBM

<!-- BEGIN:deltas -->

<!-- END:deltas -->

![Paired differences in recall at FPR 1 %](figures/paired_delta_recall.png)

![ROC curves, log-scale FPR](figures/roc_curves.png)

![Precision-recall curves](figures/pr_curves.png)

### OpenPhish live feed

Phishing-only, so only the detection rate at each model's OOD operating point is defined. The
original reported 58.5 % over URL rows; several rows share a host, so the per-host rate is also given.

<!-- BEGIN:openphish -->

<!-- END:openphish -->

## Ablations

### Fixing the threshold where a deployer would have to

Recall at 1 % FPR reads the threshold off the test set's own ROC curve — the standard definition,
and what the table above reports. A deployed model must fix its threshold in advance. Here it is
fixed on the grouped validation split at 1 % FPR and carried unchanged to the OOD set. The LightGBM
runs trained on the validation domains are left out, since for them it is not held out:

<!-- BEGIN:threshold -->

<!-- END:threshold -->

### Skipping the "already seen" filter

phishing-url-detector removed from the OOD set every Tranco host that also appears in PhiUSIIL
(about 42 % of an unfiltered sample). Leaving them in — the naive evaluation — inflates every
model, and not equally:

<!-- BEGIN:unfiltered -->

<!-- END:unfiltered -->

### What the false positives look like

The benign OOD hosts each model flags at its 1 %-FPR operating point, against the benign set as a
whole:

<!-- BEGIN:false_positives -->

<!-- END:false_positives -->

## What won, and why we think so

**The answer to the question is no.** Out of distribution, at the operating point, no small neural
model beats LightGBM. Fine-tuned DistilBERT *ties* it: its recall at 1 % FPR is statistically
indistinguishable from both the published LightGBM and the LightGBM control (both paired intervals
straddle zero). The two models trained from scratch do worse — the char-CNN clearly, the BiGRU
against the control. **In distribution, all three neural models win clearly**: on the internal
hold-out they beat LightGBM by several points with paired intervals well clear of zero, and
DistilBERT has the highest ROC-AUC on both test sets. A lead of several points in distribution
shrinks to nothing, or reverses, out of distribution. That gap is the finding.

What we think explains it — consistent with the tables above, not proven by them:

1. **The gap sits at the strict end of the ROC curve.** On ROC-AUC the neural models match or beat
   LightGBM out of distribution; DistilBERT's AUC gain even has an interval clear of zero. The
   difference is in the few hundred benign hosts each model scores *highest*: at 0.1 % FPR every
   neural model catches less phishing than LightGBM, the character models far less. On the
   log-scale ROC plot, LightGBM is far ahead below roughly 0.3 % FPR; DistilBERT catches up around
   there, the character models only at a few percent. An operational metric exposes what AUC
   averages away.
2. **The benign hosts they flag look machine-generated.** Among the benign hosts each model flags
   at 1 % FPR, the share containing digits or hyphens is several times the base rate for every
   model — highest for the character models, lower for DistilBERT, lowest for LightGBM — and
   punycode names, almost absent from the benign set, are heavily over-represented for all. In
   PhiUSIIL, random-looking strings are a strong phishing cue; in Tranco's long tail they are also
   CDN shards, trackers and internationalised names. A model that reads raw characters learns that
   cue more sharply than 51 coarse features do. Subword pretraining appears to temper it, not remove it.
3. **The character models memorise more.** Leaving already-seen hosts in the OOD set inflates the
   char-CNN and BiGRU more than LightGBM; DistilBERT is inflated about as much as LightGBM.
   Without the parent project's filter, the from-scratch models would have looked better than
   they are.
4. **Seeds matter as much as architectures at this operating point.** Re-seeding the published
   LightGBM on the same rows moves its OOD recall by about two points, and the published 70.5 % is
   the lowest of its four seeds. DistilBERT's three seeds spread even wider — one single run lands
   below the published number, another well above it. A one-seed comparison could have reported
   either a win or a loss; this is why every model has three seeds and two LightGBM references.
5. **No model keeps its promise when the threshold is fixed in advance.** Calibrated at 1 % FPR on
   held-out validation data, every model incurs several times that rate on the OOD set, and the
   neural models more than LightGBM. The headline recall figures, ours and the parent's alike,
   assume an operating point nobody could have chosen beforehand.

On the small OpenPhish probe DistilBERT detects the most phishing; with 260 hosts and no benign
counterpart, that is weak evidence either way. And matching LightGBM costs DistilBERT two orders of
magnitude more training time, a GPU, and a 67 M-parameter model where 300 trees suffice.

What this does **not** show: that neural models cannot win here. They were not tuned, saw one
dataset, and read host names only. It shows that on this evaluation, with these budgets, a model
that wins in distribution can tie or lose where it would be deployed — and that only an
out-of-distribution test at an operational metric, with several seeds, would have revealed it.

## Limitations

- **Host names only.** No path, page content, certificate or DNS signal. This is forced by the
  dataset's leaks, and it caps what any model here can learn.
- **One training dataset.** Every model learns from PhiUSIIL's hosts. A neural model with more
  capacity might benefit more from a larger, cleaner corpus than LightGBM would; this study cannot
  say.
- **One capture date.** The OOD and OpenPhish sets are the 2026-08-14 capture of live feeds.
  Another capture will give different numbers.
- **Three seeds.** Enough to see that seed variation is not negligible at this operating point,
  not enough to characterise it. Domain-level bootstrap intervals cover test-set sampling only,
  not training randomness; the per-seed column shows the latter directly.
- **No hyperparameter search.** A tuned neural model could do better. So could a tuned LightGBM.
  Neither was tuned, deliberately, so the comparison is between equally untuned models.
- **MPS non-determinism.** Neural runs on Apple-silicon GPUs are seeded but not bit-reproducible;
  re-running may move the neural numbers in the last digits. The LightGBM numbers are exact.
- **DistilBERT capped at 3 epochs** for compute (it stopped early anyway); every model's
  per-epoch validation curve is in `reports/results.json`.

## Reproduce

Python 3.12 with [uv](https://docs.astral.sh/uv/). One command from a fresh clone, with the
2026-08-14 capture of the raw data (see *Data* below for why it must be that capture):

```bash
make setup && make data COPY_FROM=<path-to>/phishing-url-detector/data/raw && make splits train eval tables test
```

| Step | What it does | Time on an M4 |
|---|---|---|
| `make data` | download or copy the raw inputs, then verify them against `data/SHA256SUMS` | seconds |
| `make splits` | build all sets with the vendored code, cached in `data/processed/` | ~10 s |
| `make train` | every model and seed; skips runs whose scores exist, so it resumes | ~1.5 h |
| `make train-fast` | everything but DistilBERT | ~15 min |
| `make eval` | gate on the reproduction, metrics, 1,000 bootstrap resamples, ablations, figures | ~2 min |
| `make tables` | rewrite every table and the headline in this README from `reports/results.json` | instant |
| `make test` | unit tests; real-data tests are skipped when the data is absent | seconds |

Every number in this README lives between `<!-- BEGIN/END -->` markers and is regenerated by
`make tables`; a test fails if the README and `reports/results.json` disagree.

## Relationship to phishing-url-detector

[phishing-url-detector](https://github.com/Yoyagm/phishing-url-detector) is the parent project:
it found the two label leaks in PhiUSIIL, defined the leak-free host-name evaluation, and shipped
an explainable LightGBM with an in-browser demo. This repository asks the next question on the
same ground. It vendors the parent's data code unchanged instead of re-implementing it, re-trains
the parent's model and checks it against the parent's published numbers before comparing anything,
and changes the evaluation in only two ways: a validation split for early stopping, and confidence
intervals with paired tests that the parent did not report.

## Data

None of the data is in this repository; `make data` fetches it and checks it against the SHA-256
of the 2026-08-14 capture. The phishing feeds rotate within hours, so a fresh download will not
match: the script says so and stops, rather than letting a different snapshot pass for the
published one.

| Source | Use | Terms |
|---|---|---|
| PhiUSIIL Phishing URL Dataset — Prasad & Chandra, *Computers & Security* 136 (2024), UCI ML Repository id 967 | training, internal hold-out | CC BY 4.0 (per the UCI page); not redistributed |
| Tranco top-1M list — Le Pochat et al., NDSS 2019 | benign side of the OOD set | see tranco-list.eu; the paper is cited; not redistributed |
| PhishTank verified online dump (Cisco Talos) | phishing side of the OOD set, by submission time | PhishTank terms of use; not redistributed |
| OpenPhish community feed | independent phishing probe | OpenPhish community-feed terms; not redistributed |
| Public Suffix List | registrable domains | MPL 2.0; frozen snapshot in `data/` |

## Dependencies

| Package | Why |
|---|---|
| `lightgbm`, `scikit-learn`, `numpy`, `pandas` | Pinned to the parent's exact versions: the baseline must reproduce its numbers to 1e-9, and the split and metric code come from these. |
| `torch` | Trains the neural models on the Apple-silicon GPU (MPS) with a CPU fallback; the de-facto standard, with no extra training framework on top. |
| `transformers` | Loads the pretrained DistilBERT and its tokenizer at a pinned revision; nothing else in the ecosystem is used. |
| `matplotlib` | Draws the figures with the parent's styling module. |
| `pytest` | Test runner (development only). |

LightGBM and PyTorch ship different OpenMP runtimes, and on macOS a process that loads both crashes
or deadlocks; the code never imports them into the same process (`src/phishsmall/neural.py`).

## Layout

```
src/phishurl/      vendored from phishing-url-detector, byte-identical (canonicalise, PSL, features, datasets)
src/phishsmall/    splits · metrics · encoding · models · training · neural · baseline · train · evaluation · figures · tables
scripts/           fetch_data.py
tests/             splits · metrics · neural models · baseline determinism · tables · vendored-file hashes
reports/           results.json — every number in this README
figures/           generated by `make eval`
MODEL_CARD.md      intended use, data, metrics, limits and risks
```

## License

MIT — see [LICENSE](LICENSE).
