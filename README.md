# Peer Review Outcome Prediction

**Can you predict whether a paper gets accepted from its text and figures?** This project uses ICLR submissions and their accept/reject decisions from [OpenReview](https://openreview.net) to test two routes:

1. **Text:** fine-tune [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) on the abstract, the full text or individual sections (Introduction, Related Work, Experiments, ...) to classify accept vs reject.
2. **Figures:** extract every figure from each PDF, filter out tables and equations with a small YOLO "janitor" classifier, then train a YOLOv8 image classifier on accept vs reject. A paper-level majority vote over its figures gives the final prediction.

University of Sheffield research project (2025), supervised by Dr Judita Preiss.

---

## Pipeline

```
OpenReview PDFs ──► VILA layout (extracted_mmda) + GROBID (extracted_grobid)
       │
       ├── TEXT ───────────────────────────────────────────────────────────────
       │   text_01_word_frequencies.py   corpus word counts (for de-hyphenation)
       │   text_02_extract_sections.py   match VILA text to GROBID section headings
       │   notebooks/text/04_merge...     join sections with decisions
       │   notebooks/text/02, 03, 05      SciBERT classifiers (full text / abstract / sections + Optuna)
       │
       └── FIGURES ────────────────────────────────────────────────────────────
           fig_01_extract_figures.py      figures from PDFs via VILA boxes (raw bitmap or rendered crop)
           ┌ janitor (removes tables / equations)
           │ fig_02_build_janitor_dataset.py   "bad" = VILA tables+equations, "good" = real figures
           │ fig_03_janitor_val_split.py       80/20 split
           │ fig_04_train_janitor.py           YOLOv8n-cls, 224 px, 5 epochs
           └ fig_05_filter_non_figures.py      delete images the janitor flags
           Experiment 1 (one composite image per paper)
             fig_07_side_composites.py / fig_08_up_down_composites.py
             fig_09_sort_by_decision.py        → accept/ reject/ folders, train/val
           Experiment 2 (one image per figure)
             fig_06_individual_dataset.py      padded 1024×1024 figures, paper-level train/val split
             fig_10_final_datasets.py          composites rebuilt from the cleaned figures
           fig_11_train_classifier.py         YOLOv8 accept/reject classifier
           fig_12_predict_majority_vote.py    paper-level vote over its figures (accept if ≥ 25% vote accept)
```

## Repository structure

```
PeerReviewOutcome/
├── data/
│   └── combined.csv          # 6,423 ICLR papers: paper_id, decision (1 = accept), abstract, title
├── notebooks/
│   ├── text/                 # PDF download + text extraction, SciBERT classifiers, merges
│   └── headings/             # development versions of the section-heading matcher
├── src/
│   ├── paths.py              # every input/output path, overridable with environment variables
│   ├── text_*.py             # text pipeline
│   ├── fig_*.py              # figure pipeline, numbered in run order
│   └── legacy/               # earlier figure extractors, kept for reference
├── requirements.txt
└── README.md
```

| Notebook | What it does |
|---|---|
| `text/01_download_pdfs_extract_text.ipynb` | Downloads PDFs from OpenReview, extracts text with PyMuPDF, merges with decisions |
| `text/02_scibert_fulltext_classifier.ipynb` | SciBERT on full paper text, 5-fold CV |
| `text/03_scibert_abstract_classifier.ipynb` | SciBERT on abstracts, k-fold CV. Reads `abstract.csv` (`Abstract`, `label`), i.e. `data/combined.csv` with columns renamed |
| `text/04_merge_sections_with_decisions.ipynb` | Joins extracted sections with `combined.csv` → `merged_papers.csv` |
| `text/05_scibert_section_classifier_optuna.ipynb` | Per-section SciBERT classifier with Optuna tuning |
| `headings/*.ipynb` | Colab development of the VILA ↔ GROBID heading matcher (final version is `src/text_02_extract_sections.py`) |

---

## Getting started

### 1. Environment

```bash
git clone https://github.com/<your-username>/PeerReviewOutcome.git
cd PeerReviewOutcome
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first.

### 2. Data

- **`data/combined.csv`** (included) is enough for the abstract classifier (`notebooks/text/03_...`). Save it as `abstract.csv` with `abstract` → `Abstract` and `decision` → `label`.
- **Everything else** needs the OpenReview corpus laid out like this (the Sheffield `preiss_group` share used this layout):

```
<OPENREVIEW_DIR>/
├── pdfs/                    # <paper_id>.pdf
├── extracted_mmda/          # VILA layout output, <paper_id>.json.gz (via the mmda library)
├── extracted_grobid/        # GROBID TEI XML
├── images_vila/             # <paper_id>.vila.zip, written by fig_01
└── 2017no_duplicates.csv    # decision file (.csv or .xlsx) with Paper_id and Decision columns
```

PDFs and decisions come from the [OpenReview API](https://docs.openreview.net/). The layout and section files come from running [VILA/mmda](https://github.com/allenai/mmda) and [GROBID](https://github.com/kermitt2/grobid) over the PDFs.

### 3. Configure paths

All paths live in `src/paths.py`. Set them with environment variables, or keep the defaults (`data/openreview`, `work/`, `runs/` under the repo root):

```bash
export OPENREVIEW_DIR=/path/to/openreview     # Windows PowerShell: $env:OPENREVIEW_DIR="X:\..."
export WORK_DIR=$PWD/work
```

### 4. Run (from the repository root)

```bash
# Text
python src/text_01_word_frequencies.py
python src/text_02_extract_sections.py

# Figures: extract, clean, build datasets, train, evaluate
python src/fig_01_extract_figures.py            # PAPER_IDS=id1,id2 to run on a subset
python src/fig_06_individual_dataset.py
python src/fig_02_build_janitor_dataset.py
python src/fig_03_janitor_val_split.py
python src/fig_04_train_janitor.py
python src/fig_05_filter_non_figures.py
python src/fig_10_final_datasets.py
python src/fig_11_train_classifier.py            # set DATA_DIR at the top to pick the dataset
python src/fig_12_predict_majority_vote.py
```

Ultralytics writes trained models to `runs/classify/<name>/weights/best.pt`. The janitor and voting scripts load them from there, so update `MODEL_PATH` if your run gets a suffix (e.g. `vila_accept_reject_classifier5`).

## Notes and caveats

- **Class imbalance:** `combined.csv` is about 70% reject (4,468) and 30% accept (1,955). Always compare accuracy with the majority-class baseline. `fig_12` now prints it.
- **Evaluation leakage (fixed):** `fig_12` used to vote over almost every paper, including ones the classifier was trained on. It now evaluates only on the papers that `fig_06` placed in the validation split.
- **Small-sample notebooks:** some notebook outputs come from tiny development subsets (e.g. the first 20 abstracts), so the fold accuracies saved in them are not results.

## Acknowledgements

Supervised by Dr Judita Preiss (University of Sheffield), who also contributed to the section-matching and figure-extraction code (see the commit history). Built on SciBERT (AllenAI), VILA/mmda (AllenAI), GROBID, PyMuPDF and Ultralytics YOLOv8.

## Author

**Duncan Alexander**, BSc Data Science, University of Sheffield
