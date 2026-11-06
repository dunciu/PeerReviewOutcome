"""
Central place for every path the pipeline uses. Override any of them with environment variables:

    OPENREVIEW_DIR  shared OpenReview corpus: pdfs/, extracted_mmda/ (VILA), extracted_grobid/,
                    images_vila/ (zipped figures per paper) and the 2017no_duplicates decision file
    WORK_DIR        where intermediate image datasets are written
    RUNS_DIR        where Ultralytics writes trained models (runs/classify/<name>/weights/best.pt)

Defaults are relative to the repository root, so run the scripts from there.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENREVIEW_DIR = os.environ.get("OPENREVIEW_DIR", os.path.join(REPO_ROOT, "data", "openreview"))
WORK_DIR = os.environ.get("WORK_DIR", os.path.join(REPO_ROOT, "work"))
RUNS_DIR = os.environ.get("RUNS_DIR", os.path.join(REPO_ROOT, "runs"))
