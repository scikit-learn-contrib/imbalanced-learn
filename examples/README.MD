# Text Sentiment (SVM) with Class Rebalancing — imbalanced-learn Example

**What:** A small, runnable demo for 3-class sentiment (negative / neutral / positive) using:

```
TF-IDF → RandomUnderSampler → LinearSVC
```

**Why:** Text features are sparse (TF-IDF). Oversampling methods like SMOTE target dense/continuous data;

**under-sampling** works out-of-the-box for sparse text.

---

## Files

* `examples/text_sentiment_svm_with_resampling.py` — example script (CLI)
* `imblearn/tests/test_text_sentiment_example.py` & `..._cli.py` — fast smoke and unit tests

---

## Setup

```bash
# in a virtual env
pip install -e .          # install this repo
pip install datasets matplotlib pytest
# optional: keep dataset cache local
export HF_DATASETS_CACHE="$PWD/.hf_cache"
```

## Run

```bash
python examples/text_sentiment_svm_with_resampling.py --plot --max-samples 6000
```

**Outputs**

* Prints **balanced accuracy** + **classification report**
* Saves `confmat_svm_imblearn.png` when `--plot` is used

**CLI options**

```
--max-samples  INT   Limit training size (None = full). Default: 6000
--plot               Save confusion matrix image
--output PATH        Image path (default: confmat_svm_imblearn.png)
```

---

## Tests

```bash
pytest -q imblearn/tests/test_text_sentiment_example.py
pytest -q imblearn/tests/test_text_sentiment_example_cli.py
```

Tests are quick, deterministic, and skipped if `datasets` isn’t installed.

---

## Notes

* Metric focus: **balanced accuracy** & **macro-F1** (better for imbalance)
* Reproducible: fixed `random_state`, controllable `--max-samples`
* Troubleshooting: low disk? use `pip --no-cache-dir`, clear caches, keep only one env active

---
