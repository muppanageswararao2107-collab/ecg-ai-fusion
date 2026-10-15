# Interpretable Deep Learning for Automated ECG Abnormality Detection
### CNN vs LSTM vs Transformer, plus two fusion strategies, on PTB-XL — with explanations and a website

This project trains three deep-learning models on the PTB-XL 12-lead ECG dataset, combines them in two
ways, explains their predictions, and serves everything through a Streamlit website that produces a PDF
report.

> **Status of this package.** The code is complete and tested, but **no trained models or results are
> included**. Every number the website shows comes from files that *your* training run creates. Until
> then the website says "Model not trained yet" rather than showing anything made up.

---

## What is compared

| Model | What it is good at | Where |
|---|---|---|
| **CNN** (1D ResNet) | local waveform shape — QRS width, ST segment, T wave | `ecgai/models.py` `CNN1D` |
| **LSTM** (bidirectional) | order and rhythm across the 10 s | `BiLSTMNet` |
| **Transformer** | relations between distant parts of the trace | `ECGTransformer` |
| **Late-fusion ensemble** | weighted average of the three trained models; weights chosen on validation | `scripts/evaluate_all.py` |
| **Parallel fusion network** | all three as branches of one network, trained together | `ParallelFusionNet` |

### How the parallel fusion network works
```
                 ┌─ CNN branch ──────────┐
 12-lead ECG ────┼─ BiLSTM branch ───────┼──► project to 128-d ──► gate (softmax over 3 branches, per ECG)
 (1000 × 12)     └─ Transformer branch ──┘                              │
                         │ (auxiliary head per branch)                   ▼
                         └──────────────► extra training loss     weighted sum ──► head ──► 5 probabilities
```
* The three branches receive the **same ECG at the same time** (parallel), each producing an embedding.
* A learned **gate** gives each branch a weight for *this* ECG (weights sum to 1). The website shows them,
  e.g. "CNN 45 %, LSTM 20 %, Transformer 35 %".
* **Deep supervision:** each branch also has its own output head during training (weight 0.3), so no
  branch stops learning just because another one dominates.
* **Branch dropout:** for 10 % of training samples one branch is hidden, so the network cannot rely on a
  single branch.

**Be honest about fusion in your viva.** It has roughly three times the parameters of a single model and
is *not guaranteed* to win. The evaluation table compares it fairly (same data, same protocol, mean ± SD
over 3 seeds) against each single model **and** against the simpler late-fusion ensemble. If the ensemble
or a single model scores the same, that is a valid finding: report it.

---

## Task and protocol (fixed, no leakage)

* **Data:** PTB-XL v1.0.3, 100 Hz, 10 s, 12 leads → input `1000 × 12`.
* **Labels:** 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP), multi-label. Records with no
  superclass are dropped and the number is reported.
* **Splits:** official `strat_fold`. Folds 1–8 train, fold 9 validation, fold 10 test. A patient never
  appears in two splits; the code asserts this.
* **Normalization:** per-lead mean/std fitted on **training folds only**, saved, and reused for every
  upload. It is never refitted on test data or uploads.
* **Validation fold only** is used for early stopping, checkpoint choice, decision thresholds, ensemble
  weights and choosing which seed the website uses. **Fold 10 is only used for the final numbers.**
* **Same protocol for every model:** BCE loss, AdamW, LR halving on plateau, gradient clipping, same light
  augmentation, same batch size, same epoch budget, early stopping on validation macro-AUROC.
* **Metrics:** macro-AUROC (primary) with 95 % bootstrap CI, per-class AUROC, macro-F1, per-class
  precision/recall/F1, confusion matrices, parameter count, training time, inference time per ECG.
* **Reference point only:** Strodthoff et al. (2021, IEEE JBHI) report macro-AUROC around 0.93 for this
  superdiagnostic task. Never present their numbers as yours.

## Explainability
* **Integrated Gradients** for all models (one common method keeps the comparison fair).
* **Grad-CAM** for the CNN as a second view.
* **Deletion test:** hide the 10 % of time each explanation marks as most important and check the model's
  probability drops more than when hiding random 10 % (Wilcoxon test). This checks the explanations are
  not arbitrary; results that do not beat random are reported as such.

---

## Quick start on macOS

All commands run in Terminal from the project folder.

```bash
cd ~/Downloads/ecg-ai-fusion          # wherever you unzipped it
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/smoke_test.py          # ~1 min: builds, trains, saves, explains every model on random tensors
python -m pytest -q                   # full test suite
```
Apple Silicon Macs train on the GPU through PyTorch `mps` automatically. Intel Macs use the CPU (slow —
use Colab instead).

### 1. Get PTB-XL (only the 100 Hz part is needed)
```bash
pip install awscli
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data/ptb-xl --exclude "records500/*"
```
Re-run the same command if it stops; it resumes. Alternatively unzip PhysioNet's zip so that
`data/ptb-xl/ptbxl_database.csv` exists (or pass `--ptbxl-dir /your/path` in the next step).

### 2. Prepare data
```bash
python scripts/prepare_data.py
```
Writes `data/cache/` (≈1 GB), `artifacts/norm.json` and `results/data_summary.json` (split sizes,
label counts, dropped records, patient-overlap check).

### 3. Train
```bash
python scripts/train_all.py --time-test   # 1 epoch per model; prints a worst-case time estimate
python scripts/train_all.py               # all 4 models × 3 seeds
```
If the estimate is too long for your Mac, use **Colab** (next section), or train one seed first:
`python scripts/train_all.py --seeds 0`, and state in your report how many seeds were used.

### 4. Evaluate and explain
```bash
python scripts/evaluate_all.py   # tables, figures, ensemble, artifacts/deployment.json
python scripts/run_xai.py        # IG / Grad-CAM figures + deletion test
```

### 5. Run the website
```bash
streamlit run app.py
```
It opens at http://localhost:8501.

---

## Training on Google Colab (free GPU) instead

1. Upload `ecg-ai-fusion.zip` to **My Drive**.
2. Open `notebooks/colab_train.ipynb` in Colab, choose **Runtime → Change runtime type → T4 GPU**, and run
   the cells top to bottom. Each seed is copied to Drive as soon as it finishes.
3. Download `ecg_trained_outputs.zip` from Drive and unzip it **inside** the project folder on your Mac.
   It contains `artifacts/` (models), `results/` (tables, figures) and a test-fold copy of the data so the
   website's demo picker works without downloading PTB-XL again.
4. `streamlit run app.py`

---

## Using the website

1. **Analyze ECG**
   * *PTB-XL test record* — pick a held-out fold-10 record (filter by reference label) and press
     **Load record**. Reference labels are shown next to the predictions.
   * *Upload WFDB* — select the `.hea` **and** `.dat` files together.
   * *Upload CSV* — header row with the 12 lead names, one row per sample; enter the sampling rate and
     units (mV or µV).
   The preview shows quality checks (missing values, flat leads, clipping, amplitude, noise), an automated
   heart-rate estimate and the 12-lead trace. Press **Run analysis with all models**.
2. **Results** — probabilities from all five models side by side, which classes each model flags (using
   thresholds tuned on validation), the fusion gate weights, and **Compute explanation** for any model and
   class.
3. **Report** — **Create PDF report**, then **Download PDF report**. JSON results and the processed
   100 Hz signal can also be downloaded.
4. **Model comparison** — everything from `results/`, loaded from files.

**Upload rules.** Only complete 12-lead recordings are accepted; single-lead or reduced-lead files are
rejected with an explanation (missing leads are never invented). Recordings shorter than 10 s are
rejected. Longer ones are cut into 10 s windows (up to 5 minutes) and the probabilities are averaged.
Uploads are written to a temporary folder only while being parsed and deleted immediately.

---

## Project structure
```
ecg-ai-fusion/
├── app.py                     Streamlit website
├── config.yaml                every setting in one place
├── requirements.txt
├── ecgai/
│   ├── data.py                PTB-XL labels, official folds, cache, patient-overlap checks
│   ├── preprocess.py          lead names, resampling, windows, normalization
│   ├── wfdb_io.py             WFDB reader/writer (formats 16, 212, 80)
│   ├── upload.py              WFDB/CSV upload validation → model input
│   ├── quality.py             signal-quality checks
│   ├── measures.py            heart-rate estimate (labelled as automated estimate)
│   ├── models.py              CNN, BiLSTM, Transformer, ParallelFusionNet
│   ├── train.py               shared training loop, checkpoints
│   ├── metrics.py             AUROC, thresholds, F1, bootstrap CI, ensemble weights
│   ├── inference.py           loads deployed models, runs them in parallel threads
│   ├── xai.py                 Integrated Gradients, Grad-CAM, deletion test
│   ├── plotting.py            ECG-paper plots
│   └── report.py              PDF report
├── scripts/                   smoke_test, prepare_data, train_all, evaluate_all, run_xai
├── notebooks/colab_train.ipynb
├── tests/                     pytest suite
├── .streamlit/config.toml     theme and 20 MB upload limit
└── data/ artifacts/ results/  created by the scripts
```

## What was verified before delivery
* The non-PyTorch suite (labels, folds, patient overlap, WFDB/CSV parsing, resampling, quality checks,
  heart-rate estimate, normalization, metrics, ensemble weights, PDF report, temporary-file cleanup):
  **52 tests passing**.
* `prepare_data.py` run end to end on a miniature folder with PTB-XL's exact file layout.
* Every website page exercised with stand-in Streamlit modules, including the PDF download.
* **Not run before delivery:** the PyTorch parts (models, training, XAI, inference) and the real Streamlit
  rendering, because PyTorch could not be installed in the build environment. They have their own tests
  (`tests/test_models_torch.py`) and `scripts/smoke_test.py`. **Run both first.** If anything fails, the
  error message will point at the exact line.

## Limitations
* One dataset, recorded 1989–1996 with one manufacturer's devices; no external validation.
* Five broad superclasses only — not specific diagnoses, severity or urgency.
* Probabilities are not calibrated clinical risks.
* Explanations show model behaviour, which may differ from clinical reasoning.
* The heart-rate estimate is a simple detector and is not validated.

*Research prototype for educational purposes. Not a medical device and not a substitute for evaluation by
a qualified clinician.*

## Citations
* Wagner P., Strodthoff N., Bousseljot R., Samek W., Schaeffter T. (2022). *PTB-XL, a large publicly
  available electrocardiography dataset* (version 1.0.3). PhysioNet.
* Strodthoff N., Wagner P., Schaeffter T., Samek W. (2021). *Deep learning for ECG analysis: benchmarks and
  insights from PTB-XL.* IEEE Journal of Biomedical and Health Informatics 25(5).
* Sundararajan M., Taly A., Yan Q. (2017). *Axiomatic attribution for deep networks* (Integrated Gradients).
* Selvaraju R. R. et al. (2017). *Grad-CAM.*
