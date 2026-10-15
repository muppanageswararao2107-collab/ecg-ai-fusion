"""ECG·AI — interpretable ECG abnormality detection (research prototype).

Run:  streamlit run app.py
"""
from __future__ import annotations

import inspect
import io
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ecgai.config import load_config, path as cfg_path
from ecgai.inference import ModelBundle, ModelsNotTrained, check_ready
from ecgai.plotting import fig_to_png, plot_12lead
from ecgai.preprocess import STANDARD_LEADS, load_normalization
from ecgai.quality import FAIL, PASS, WARN
from ecgai.report import DISCLAIMER, build_pdf
from ecgai.upload import (SUPPORTED_HELP, ECGInput, UploadError, parse_csv_upload, parse_wfdb_upload,
                          prepare_for_model)

st.set_page_config(page_title="ECG·AI – interpretable ECG analysis", page_icon="🫀", layout="wide")

CFG = load_config()
CLASSES = CFG["data"]["classes"]
FS = int(CFG["data"]["sampling_rate"])
LENGTH = int(CFG["data"]["length"])
MAX_BYTES = int(CFG["app"]["max_upload_mb"]) * 1024 * 1024
RESULTS = cfg_path(CFG, "results_dir")
MODEL_ORDER = ["cnn", "lstm", "transformer", "ensemble", "fusion"]
NAMES = {"cnn": "CNN", "lstm": "LSTM", "transformer": "Transformer", "ensemble": "Ensemble (late fusion)",
         "fusion": "Parallel fusion"}
LONG_NAMES = {"cnn": "CNN (1D ResNet)", "lstm": "LSTM (BiLSTM)", "transformer": "Transformer",
              "ensemble": "Late-fusion ensemble", "fusion": "Parallel fusion (CNN+LSTM+Transformer)"}
CLASS_INFO = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "STTC": "ST/T change",
              "CD": "Conduction disturbance", "HYP": "Hypertrophy"}
COLORS = {"cnn": "#0E6E6A", "lstm": "#2563EB", "transformer": "#7C3AED", "ensemble": "#B45309", "fusion": "#B42318"}
PAGES = ["Overview", "Analyze ECG", "Results", "Model comparison", "Report", "About & limitations"]

# ----------------------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, .stTextInput, .stSelectbox, .stRadio {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
:root { --ink:#1B2430; --teal:#0E6E6A; --grid:#E8A3A3; --grid-soft:#F6D6D6; --alarm:#B42318;
        --caution:#B45309; --ok:#15803D; --line:#DDE3E8; --muted:#5B6673; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }
h1 { font-weight: 700; font-size: 2.1rem; line-height: 1.15; }
h2 { font-weight: 600; font-size: 1.35rem; margin-top: 1.4rem; }
h3 { font-weight: 600; font-size: 1.05rem; }
.paper {
  position: relative; border: 1px solid var(--grid); border-radius: 6px; padding: 28px 30px 26px;
  background-color: #FFFCFC;
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px), linear-gradient(90deg, var(--grid) 1px, transparent 1px),
    linear-gradient(var(--grid-soft) 1px, transparent 1px), linear-gradient(90deg, var(--grid-soft) 1px, transparent 1px);
  background-size: 50px 50px, 50px 50px, 10px 10px, 10px 10px;
  margin-bottom: 1.2rem;
}
.paper h1 { margin: 0 0 6px 0; background: #FFFCFC; display: inline-block; padding: 2px 8px; }
.paper p { background: #FFFCFC; display: inline-block; padding: 4px 8px; margin: 0; max-width: 66ch;
           color: var(--ink); font-size: 1.02rem; line-height: 1.5; }
.step { border-left: 3px solid var(--teal); padding: 2px 0 2px 12px; margin: 6px 0 14px; }
.step b { color: var(--teal); }
.pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 600; }
.pill.ok { background:#E7F5EC; color: var(--ok); }
.pill.warn { background:#FDF1E3; color: var(--caution); }
.pill.fail { background:#FDECEA; color: var(--alarm); }
.pill.idle { background:#EEF1F4; color: var(--muted); }
.note { border: 1px solid var(--line); background: #FFFFFF; border-radius: 6px; padding: 12px 14px;
        color: var(--muted); font-size: 0.9rem; line-height: 1.5; }
.disclaimer { border-left: 4px solid var(--caution); background: #FFF8EE; padding: 10px 14px; border-radius: 4px;
              font-size: 0.9rem; color: #6B3A06; }
.arch { display:grid; grid-template-columns: 1fr 1.3fr 1fr 1fr; gap: 10px; align-items: center; margin: 8px 0 4px; }
.arch .box { border:1px solid var(--line); background:#fff; border-radius:6px; padding:10px 12px; font-size:0.88rem; }
.arch .branch { border-left: 4px solid var(--teal); margin-bottom: 6px; }
.arch .gate { border: 1.5px solid var(--alarm); }
.tnum { font-variant-numeric: tabular-nums; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-size: 1.35rem; }
@media (max-width: 760px) { .arch { grid-template-columns: 1fr; } .paper { padding: 18px 14px; } }
</style>
""", unsafe_allow_html=True)


def _width_kwargs(fn) -> dict:
    """Stretch a chart/table to the container across old and new Streamlit versions."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params and isinstance(params["width"].default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def show_plot(fig):
    st.plotly_chart(fig, **_width_kwargs(st.plotly_chart))


def show_table(df, **kw):
    st.dataframe(df, hide_index=True, **_width_kwargs(st.dataframe), **kw)


def pill(status: str) -> str:
    cls = {PASS: "ok", WARN: "warn", FAIL: "fail"}.get(status, "idle")
    label = {PASS: "Pass", WARN: "Warning", FAIL: "Fail"}.get(status, status)
    return f'<span class="pill {cls}">{label}</span>'


# ----------------------------------------------------------------------------- cached resources
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Loading models …")
def _load_bundle(_stamp: float):
    try:
        return ModelBundle(CFG, device="cpu"), "Models loaded."
    except ModelsNotTrained as exc:
        return None, str(exc)
    except Exception as exc:  # e.g. corrupted checkpoint
        return None, f"Models could not be loaded: {exc}"


def get_bundle():
    ok, msg = check_ready(CFG)
    if not ok:
        return None, msg
    return _load_bundle(_mtime(cfg_path(CFG, "artifacts_dir") / "deployment.json"))


def get_norm():
    p = cfg_path(CFG, "artifacts_dir") / "norm.json"
    return load_normalization(p) if p.exists() else None


@st.cache_resource(show_spinner="Loading ECG data …")
def _load_cache(_stamp: float):
    """Load local prepared data or the private Hugging Face demo cache."""
    cache = cfg_path(CFG, "cache_dir")

    for x_name, m_name in [
        ("X.npy", "meta.csv"),
        ("test_X.npy", "test_meta.csv"),
    ]:
        x_path, m_path = cache / x_name, cache / m_name
        if x_path.exists() and m_path.exists():
            X = np.load(x_path, mmap_mode="r")
            meta = pd.read_csv(m_path)
            if len(meta) == X.shape[0]:
                return X, meta.reset_index(drop=True)

    try:
        from huggingface_hub import hf_hub_download

        token = st.secrets["HF_TOKEN"]
        repo_id = st.secrets["HF_DATASET_REPO"]

        x_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="test_X.npy",
            token=token,
        )
        m_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="test_meta.csv",
            token=token,
        )

        X = np.load(x_path, mmap_mode="r")
        meta = pd.read_csv(m_path)

        if len(meta) != X.shape[0]:
            return None, None
        if "split" not in meta.columns or not (meta["split"] == "test").all():
            return None, None

        return X, meta.reset_index(drop=True)

    except Exception as exc:
        print(f"Could not load the PTB-XL demo cache: {exc}")
        return None, None


def get_cache():
    cache = cfg_path(CFG, "cache_dir")
    return _load_cache(_mtime(cache / "meta.csv") + _mtime(cache / "test_meta.csv"))


def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- plots
def ecg_figure(signal_mv: np.ndarray, fs: float, leads: list[str], attribution: np.ndarray | None = None,
               height_per_lead: int = 62) -> go.Figure:
    n = signal_mv.shape[0]
    t = np.arange(n) / fs
    fig = make_subplots(rows=len(leads), cols=1, shared_xaxes=True, vertical_spacing=0.004)
    att = None
    if attribution is not None:
        att = np.abs(attribution)
        att = att / att.max() if att.max() > 0 else att
    scatter = go.Scattergl if n > 5000 else go.Scatter
    for j, lead in enumerate(leads):
        y = signal_mv[:, j]
        lo, hi = float(min(-1.0, y.min() - 0.2)), float(max(1.0, y.max() + 0.2))
        if att is not None:
            fig.add_trace(go.Heatmap(z=[att[:, j], att[:, j]], x=t, y=[lo, hi], zmin=0, zmax=1, showscale=False,
                                     colorscale=[[0, "rgba(255,255,255,0)"], [1, "rgba(180,35,24,0.65)"]],
                                     hoverinfo="skip"), row=j + 1, col=1)
        fig.add_trace(scatter(x=t, y=y, mode="lines", line=dict(color="#1B2430", width=1), name=lead,
                              hovertemplate=f"{lead}: %{{y:.2f}} mV at %{{x:.2f}} s<extra></extra>"),
                      row=j + 1, col=1)
        fig.update_yaxes(title_text=lead, title_standoff=4, range=[lo, hi], row=j + 1, col=1)
    fig.update_xaxes(dtick=0.2, gridcolor="#F2C4C4", showgrid=True, zeroline=False, showticklabels=False)
    fig.update_xaxes(showticklabels=True, title_text="Time (s)", row=len(leads), col=1)
    fig.update_yaxes(dtick=0.5, gridcolor="#F2C4C4", showgrid=True, zeroline=False, showticklabels=False,
                     title_font=dict(size=11))
    fig.update_layout(height=height_per_lead * len(leads) + 50, showlegend=False, plot_bgcolor="#FFFCFC",
                      paper_bgcolor="#FFFFFF", margin=dict(l=48, r=12, t=8, b=36), hovermode="x unified")
    return fig


def prob_figure(models: dict) -> go.Figure:
    fig = go.Figure()
    for m in MODEL_ORDER:
        if m in models:
            fig.add_trace(go.Bar(name=NAMES[m], x=CLASSES, y=[models[m]["probs"][c] for c in CLASSES],
                                 marker_color=COLORS[m],
                                 hovertemplate=f"{NAMES[m]}<br>%{{x}}: %{{y:.2f}}<extra></extra>"))
    fig.update_layout(barmode="group", height=320, yaxis=dict(range=[0, 1], title="Model probability"),
                      plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", legend=dict(orientation="h", y=1.12),
                      margin=dict(l=50, r=10, t=30, b=30))
    return fig


# ----------------------------------------------------------------------------- state
def reset_analysis(keep_input: bool = False):
    for k in ["analysis", "xai", "report_id", "pdf"] + ([] if keep_input else ["ecg_input", "prepared"]):
        st.session_state.pop(k, None)


def set_input(inp: ECGInput):
    reset_analysis()
    norm = get_norm() or {"sampling_rate": FS, "length": LENGTH, "mean": np.zeros(12, np.float32),
                          "std": np.ones(12, np.float32), "preview_only": True}
    try:
        prepared = prepare_for_model(inp, norm, int(CFG["app"]["max_windows"]))
    except UploadError as exc:
        st.error(str(exc))
        return
    st.session_state.ecg_input = inp
    st.session_state.prepared = prepared
    st.session_state.preview_only = bool(norm.get("preview_only"))


def go_to(page: str):
    st.session_state.nav_target = page
    st.rerun()


# ----------------------------------------------------------------------------- sidebar
if "nav_target" in st.session_state:
    st.session_state.nav = st.session_state.pop("nav_target")

with st.sidebar:
    st.markdown("### 🫀 ECG·AI")
    st.caption("Interpretable ECG abnormality detection — research prototype")
    page = st.radio("Go to", PAGES, key="nav", label_visibility="collapsed")
    st.divider()
    bundle, bundle_msg = get_bundle()
    X_cache, meta_cache = get_cache()
    st.markdown(f"Models {pill(PASS) if bundle else pill('Not trained')}", unsafe_allow_html=True)
    st.markdown(f"PTB-XL data {pill(PASS) if meta_cache is not None else pill('Not prepared')}",
                unsafe_allow_html=True)
    st.markdown(f"Evaluation {pill(PASS) if (RESULTS / 'comparison.csv').exists() else pill('Not run')}",
                unsafe_allow_html=True)
    if "ecg_input" in st.session_state:
        st.divider()
        st.caption(f"Current ECG: {st.session_state.ecg_input.filename}")
        if st.button("Clear current ECG"):
            reset_analysis()
            st.rerun()
    st.divider()
    st.caption("Uploads are processed in memory and deleted after parsing. Nothing is stored.")


# ----------------------------------------------------------------------------- pages
def page_overview():
    st.markdown("""
<div class="paper">
  <h1>Interpretable ECG abnormality detection</h1><br>
  <p>Analyse a 12-lead ECG with a CNN, an LSTM and a Transformer running in parallel, compare their outputs,
  combine them, and see which parts of the trace drove each prediction.</p>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("## How it works")
        for i, (title, text) in enumerate([
            ("Load an ECG", "Pick a held-out PTB-XL test record, or upload a 12-lead WFDB or CSV file."),
            ("Check the signal", "Missing values, flat leads, clipping, amplitude and noise are checked; "
                                 "the signal is resampled to 100 Hz and cut into 10 s windows."),
            ("Run all models in parallel", "Three standalone models, a late-fusion ensemble and the parallel "
                                           "fusion network each score five diagnostic superclasses."),
            ("Explain and report", "Integrated Gradients highlights influential regions; a PDF report "
                                   "records everything that was computed."),
        ], start=1):
            st.markdown(f'<div class="step"><b>{i}. {title}</b><br>{text}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("## What the models detect")
        show_table(pd.DataFrame({"Class": list(CLASS_INFO), "Meaning": list(CLASS_INFO.values())}))
        st.caption("A recording can belong to several classes at once (multi-label).")

    st.markdown("## The parallel fusion network")
    st.markdown("""
<div class="arch">
  <div class="box"><b>12-lead ECG</b><br>10 s at 100 Hz, standardized with training statistics</div>
  <div>
    <div class="box branch"><b>CNN</b> — local waveform shape (QRS, ST segment)</div>
    <div class="box branch" style="border-left-color:#2563EB"><b>BiLSTM</b> — order and rhythm over time</div>
    <div class="box branch" style="border-left-color:#7C3AED"><b>Transformer</b> — relations across the whole 10 s</div>
  </div>
  <div class="box gate"><b>Learned gate</b><br>weights each branch for this ECG (weights sum to 1)</div>
  <div class="box"><b>Fusion head</b><br>5 probabilities: NORM, MI, STTC, CD, HYP</div>
</div>""", unsafe_allow_html=True)
    st.markdown(
        '<div class="note">All three branches see the same ECG at the same time. Each branch also has its own '
        "auxiliary output during training so none of them stops learning, and one branch is randomly hidden "
        "for 10% of training samples so the network cannot lean on a single branch. For comparison, the app "
        "also shows a late-fusion ensemble: a weighted average of the three separately trained models, with "
        "weights chosen on the validation fold.</div>", unsafe_allow_html=True)

    st.markdown("## Status")
    s1, s2, s3 = st.columns(3)
    s1.markdown(f"**Models** {pill(PASS) if bundle else pill('Not trained')}<br>"
                f"<span style='color:#5B6673;font-size:.88rem'>{bundle_msg}</span>", unsafe_allow_html=True)
    s2.markdown(f"**PTB-XL data** {pill(PASS) if meta_cache is not None else pill('Not prepared')}<br>"
                f"<span style='color:#5B6673;font-size:.88rem'>"
                f"{f'{len(meta_cache):,} records available' if meta_cache is not None else 'Run scripts/prepare_data.py'}"
                "</span>", unsafe_allow_html=True)
    s3.markdown(f"**Evaluation** {pill(PASS) if (RESULTS / 'comparison.csv').exists() else pill('Not run')}<br>"
                "<span style='color:#5B6673;font-size:.88rem'>See Model comparison</span>", unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer" style="margin-top:18px">{DISCLAIMER}</div>', unsafe_allow_html=True)


def page_analyze():
    st.markdown("# Analyze an ECG")
    tab_ptb, tab_wfdb, tab_csv = st.tabs(["PTB-XL test record", "Upload WFDB", "Upload CSV"])

    with tab_ptb:
        if meta_cache is None:
            st.info("PTB-XL has not been prepared yet. Run `python scripts/prepare_data.py`, then reload this page.")
        else:
            test = meta_cache[meta_cache["split"] == "test"]
            st.caption(f"{len(test):,} records from PTB-XL fold 10. The models never saw these during training, "
                       "threshold tuning or model selection.")
            c1, c2 = st.columns([1, 2])
            cls = c1.selectbox("Reference label", ["Any"] + CLASSES)
            subset = test if cls == "Any" else test[test[cls] == 1]
            ids = subset["ecg_id"].tolist()[:500]

            def fmt(eid):
                row = subset[subset["ecg_id"] == eid].iloc[0]
                return f"ecg_id {eid} — " + (", ".join(c for c in CLASSES if row[c] == 1) or "none")

            choice = c2.selectbox("Record", ids, format_func=fmt) if ids else None
            if choice is not None and st.button("Load record", type="primary"):
                pos = int(np.where(meta_cache["ecg_id"].to_numpy() == choice)[0][0])
                row = meta_cache.iloc[pos]
                set_input(ECGInput(signal=np.asarray(X_cache[pos], dtype=np.float32), fs=float(FS),
                                   leads=list(STANDARD_LEADS), filename=f"PTB-XL ecg_id {choice}",
                                   source="PTB-XL test record",
                                   notes=["PTB-XL 100 Hz record from the held-out test fold (10)."],
                                   ecg_id=int(choice), true_labels=[c for c in CLASSES if row[c] == 1]))

    with tab_wfdb:
        st.caption("Upload the .hea header and its .dat signal file together. All 12 standard leads are required.")
        files = st.file_uploader("WFDB files", type=["hea", "dat"], accept_multiple_files=True)
        if st.button("Check upload", key="wfdb_btn", type="primary", disabled=not files):
            try:
                set_input(parse_wfdb_upload({f.name: f.getvalue() for f in files}, MAX_BYTES))
            except UploadError as exc:
                st.error(str(exc))

    with tab_csv:
        st.caption("One column per lead with a header row (I, II, III, aVR, aVL, aVF, V1–V6). "
                   "A time column is ignored. A CSV does not store its sampling rate, so enter it below.")
        f = st.file_uploader("CSV file", type=["csv"])
        c1, c2 = st.columns(2)
        fs_val = c1.number_input("Sampling rate (Hz)", min_value=0.0, max_value=10000.0, value=0.0, step=1.0,
                                 help="Required. Common values: 100, 250, 360, 500, 1000.")
        units = c2.radio("Units", ["mV", "µV"], horizontal=True)
        if st.button("Check upload", key="csv_btn", type="primary", disabled=f is None):
            try:
                set_input(parse_csv_upload(f.name, f.getvalue(), fs_val if fs_val > 0 else None, units, MAX_BYTES))
            except UploadError as exc:
                st.error(str(exc))
    with st.expander("Supported formats"):
        st.write(SUPPORTED_HELP)
        st.write("Recordings from devices other than the PTB-XL source may differ in filtering and population; "
                 "results on them are less reliable.")

    if "ecg_input" not in st.session_state:
        return
    inp: ECGInput = st.session_state.ecg_input
    prep = st.session_state.prepared
    st.markdown("## Preview")
    m = st.columns(5)
    m[0].metric("Source", inp.source.replace("Uploaded ", ""))
    m[1].metric("Leads", len(inp.leads))
    m[2].metric("Sampling rate", f"{inp.fs:g} Hz")
    m[3].metric("Duration", f"{inp.duration_s:.1f} s")
    m[4].metric("10 s windows", int(prep.windows_norm.shape[0]))
    if inp.true_labels is not None:
        st.markdown(f"PTB-XL reference labels: **{', '.join(inp.true_labels) or 'none'}**")

    st.markdown(f"### Signal quality {pill(prep.quality_status)}", unsafe_allow_html=True)
    show_table(pd.DataFrame([{"Check": q.name, "Result": {"PASS": "✅ Pass", "WARN": "⚠️ Warning", "FAIL": "⛔ Fail"}[q.status],
                              "Details": q.message} for q in prep.quality]))
    hr = prep.heart_rate
    st.caption(f"Heart rate (automated estimate, lead {hr['lead']}): {hr['bpm']:.0f} bpm — not a validated measurement."
               if hr["available"] else f"Heart rate: not available — {hr['reason']}")
    for note in prep.notes:
        st.caption(note)

    n_show = min(prep.signal_100.shape[0], LENGTH)
    st.markdown("### First 10 seconds at 100 Hz")
    show_plot(ecg_figure(prep.signal_100[:n_show], FS, inp.leads))

    if prep.quality_status == FAIL:
        st.error("The signal failed a quality check, so it will not be analysed. Fix the issue above and upload again.")
        return
    if bundle is None:
        st.warning(f"{bundle_msg} Predictions are not shown until trained models are available.")
        return
    if st.session_state.get("preview_only"):
        st.warning("Normalization statistics are missing, so the signal cannot be prepared for the models.")
        return
    if st.button("Run analysis with all models", type="primary"):
        with st.spinner("Running CNN, LSTM, Transformer, ensemble and fusion …"):
            st.session_state.analysis = bundle.predict(prep.windows_norm)
            st.session_state.report_id = f"ECG-{uuid.uuid4().hex[:8].upper()}"
            st.session_state.xai = {}
        go_to("Results")


def page_results():
    st.markdown("# Results")
    if "analysis" not in st.session_state:
        st.info("No analysis yet. Load an ECG on the Analyze ECG page and run the models.")
        return
    inp: ECGInput = st.session_state.ecg_input
    prep = st.session_state.prepared
    res = st.session_state.analysis
    models = res["models"]
    st.caption(f"{inp.source}: {inp.filename}. Report ID {st.session_state.report_id}. "
               f"{res['n_windows']} window(s) of 10 s analysed.")
    if inp.true_labels is not None:
        st.markdown(f"PTB-XL reference labels (cardiologist annotations): **{', '.join(inp.true_labels) or 'none'}**")

    st.markdown("## Model outputs")
    order = [m for m in MODEL_ORDER if m in models]
    rows = []
    for c in CLASSES:
        row = {"Class": f"{c} — {CLASS_INFO[c]}"}
        for m in order:
            row[NAMES[m]] = models[m]["probs"][c]
        n_flag = sum(models[m]["flags"][c] for m in order)
        row["Flagged by"] = f"{n_flag} of {len(order)}"
        rows.append(row)
    df = pd.DataFrame(rows)
    col_cfg = {NAMES[m]: st.column_config.ProgressColumn(NAMES[m], min_value=0.0, max_value=1.0, format="%.2f")
               for m in order}
    show_table(df, column_config=col_cfg)
    flagged = {m: [c for c in CLASSES if models[m]["flags"][c]] for m in order}
    st.markdown("**Above each model's decision threshold:** " +
                "; ".join(f"{NAMES[m]}: {', '.join(v) if v else 'none'}" for m, v in flagged.items()))
    st.caption("Probabilities are model outputs averaged over the 10 s windows, not clinical risks. Decision "
               "thresholds were tuned on the PTB-XL validation fold only.")
    show_plot(prob_figure(models))

    if "fusion" in models and models["fusion"].get("branch_weights"):
        st.markdown("## How the fusion network weighted its branches")
        bw = models["fusion"]["branch_weights"]
        cols = st.columns(3)
        for col, b in zip(cols, ["cnn", "lstm", "transformer"]):
            col.metric(NAMES[b], f"{bw[b] * 100:.0f}%")
        st.caption("Gate weights for this ECG. They show what the fusion model relied on, not which branch is correct.")

    if res["n_windows"] > 1:
        with st.expander("Per-window probabilities"):
            m_sel = st.selectbox("Model", order, format_func=lambda m: NAMES[m], key="win_model")
            pw = np.array(res["per_window"][m_sel])
            show_table(pd.DataFrame(pw, columns=CLASSES).assign(Window=lambda d: d.index + 1)
                       .set_index("Window").reset_index().round(3))

    if bundle is None:
        st.warning(f"{bundle_msg} Explanations need the trained models.")
        return
    st.markdown("## Explanation")
    st.caption("Integrated Gradients: which time regions and leads pushed this model's output for the chosen class. "
               "The baseline is the average training ECG.")
    c1, c2, c3, c4 = st.columns(4)
    xm = c1.selectbox("Model", [m for m in ["fusion", "cnn", "lstm", "transformer"] if m in bundle.models],
                      format_func=lambda m: NAMES[m])
    default_cls = next((c for c in CLASSES if models[xm]["flags"][c]), CLASSES[0])
    xc = c2.selectbox("Class", CLASSES, index=CLASSES.index(default_cls))
    methods = ["Integrated Gradients"] + (["Grad-CAM"] if xm == "cnn" else [])
    xmeth = c3.selectbox("Method", methods)
    xw = c4.number_input("Window", min_value=1, max_value=int(res["n_windows"]), value=1, step=1)
    key = f"{xm}|{xc}|{xmeth}|{xw}"
    if st.button("Compute explanation", type="primary"):
        from ecgai.xai import grad_cam_1d, integrated_gradients
        with st.spinner("Computing attributions …"):
            x = prep.windows_norm[int(xw) - 1]
            j = CLASSES.index(xc)
            if xmeth == "Grad-CAM":
                cam = grad_cam_1d(bundle.models["cnn"], x, j)
                attr, delta = np.repeat(cam[:, None], 12, axis=1), None
            else:
                ig = integrated_gradients(bundle.models[xm], x, j, steps=int(CFG["xai"]["ig_steps"]))
                attr, delta = ig["attr"], ig["delta"]
        st.session_state.xai[key] = {"attr": attr, "delta": delta, "model": xm, "class": xc, "method": xmeth,
                                     "window": int(xw)}
        st.session_state.xai_last = key
    if key in st.session_state.get("xai", {}):
        e = st.session_state.xai[key]
        start = (e["window"] - 1) * LENGTH
        raw = prep.signal_100[start:start + LENGTH]
        show_plot(ecg_figure(raw, FS, inp.leads, attribution=e["attr"]))
        lead_imp = np.abs(e["attr"]).sum(axis=0)
        top = [inp.leads[i] for i in np.argsort(lead_imp)[::-1][:3]]
        st.caption(f"Most influential leads for {NAMES[e['model']]} → {e['class']}: {', '.join(top)}. "
                   + (f"Integrated Gradients completeness error: {e['delta']:+.3f} (closer to 0 is better). "
                      if e["delta"] is not None else "")
                   + "Highlighted regions describe model behaviour, not clinical findings.")
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


def page_comparison():
    st.markdown("# Model comparison")
    comp_p = RESULTS / "comparison.csv"
    data_summary = read_json(RESULTS / "data_summary.json")
    if data_summary:
        st.markdown("## Data")
        sizes = data_summary["split_sizes"]
        c = st.columns(4)
        c[0].metric("Records used", f"{data_summary['records_used']:,}")
        c[1].metric("Train (folds 1–8)", f"{sizes['train']:,}")
        c[2].metric("Validation (fold 9)", f"{sizes['val']:,}")
        c[3].metric("Test (fold 10)", f"{sizes['test']:,}")
        st.caption(f"{data_summary['records_without_superclass_dropped']} records without a diagnostic superclass "
                   f"were dropped. Patient-overlap check between splits: {data_summary['patient_overlap_check']}."
                   + (" LIMIT MODE: only part of PTB-XL was used." if data_summary.get("limit_mode") else ""))
        show_table(pd.DataFrame(data_summary["label_counts"]).T.reset_index().rename(columns={"index": "Split"}))
    if not comp_p.exists():
        st.info("No evaluation results yet. Run `python scripts/train_all.py` and then "
                "`python scripts/evaluate_all.py`. Nothing on this page is ever filled in by hand.")
        return
    comp = pd.read_csv(comp_p)
    st.markdown("## Test-fold performance")
    view = pd.DataFrame({
        "Model": comp["display_name"],
        "Seeds": comp["n_seeds"],
        "Test macro-AUROC": [f"{m:.4f}" + (f" ± {s:.4f}" if pd.notna(s) else "")
                             for m, s in zip(comp["test_macro_auroc_mean"], comp["test_macro_auroc_std"])],
        "Test macro-F1": [f"{m:.4f}" + (f" ± {s:.4f}" if pd.notna(s) else "")
                          for m, s in zip(comp["test_macro_f1_mean"], comp["test_macro_f1_std"])],
        "Val macro-AUROC": comp["val_macro_auroc_mean"].round(4),
        "Parameters": comp["n_params"].map(lambda v: f"{int(v):,}"),
        "ms per ECG": comp["inference_ms_per_ecg"].round(1),
    })
    show_table(view)
    st.caption("Mean ± standard deviation over training seeds. Thresholds for F1 were tuned on the validation fold. "
               "For reference only: Strodthoff et al. (2021) report macro-AUROC around 0.93 for this task "
               "with their models; those are not this project's numbers.")
    figs = RESULTS / "figures"
    for name, caption in [("auroc_comparison.png", "Test macro-AUROC by model"),
                          ("roc_curves.png", "ROC curves per class (deployed seed of each model)"),
                          ("training_curves.png", "Validation macro-AUROC during training"),
                          ("fusion_branch_weights.png", "Average fusion gate weights by true class (test fold)")]:
        if (figs / name).exists():
            st.markdown(f"### {caption}")
            st.image(str(figs / name))
    pc = RESULTS / "per_class.csv"
    if pc.exists():
        st.markdown("## Per-class results")
        show_table(pd.read_csv(pc).round(4))
    with st.expander("Confusion matrices (test fold)"):
        for m in ["cnn", "lstm", "transformer", "fusion"]:
            p = figs / f"confusion_{m}.png"
            if p.exists():
                st.image(str(p))
    dsum = RESULTS / "xai" / "deletion_test_summary.csv"
    st.markdown("## Are the explanations meaningful?")
    if dsum.exists():
        d = pd.read_csv(dsum)
        d = d[d["class"] == "ALL"] if "ALL" in set(d["class"]) else d
        show_table(d.round(4))
        st.caption("Deletion test: hiding the 10% of time each explanation marks as most important should lower the "
                   "model's probability more than hiding random 10% of time. A Wilcoxon p-value below 0.05 means "
                   "the difference is unlikely to be chance.")
        ex = sorted((RESULTS / "xai").glob("ig_*_*.png"))
        if ex:
            with st.expander("Example explanation figures"):
                pick = st.selectbox("Figure", [p.name for p in ex])
                st.image(str(RESULTS / "xai" / pick))
    else:
        st.info("Run `python scripts/run_xai.py` to compute the deletion test and example figures.")


def page_report():
    st.markdown("# Report")
    if "analysis" not in st.session_state:
        st.info("Analyse an ECG first; the report is built from that analysis.")
        return
    inp: ECGInput = st.session_state.ecg_input
    prep = st.session_state.prepared
    res = st.session_state.analysis
    xai_entries = st.session_state.get("xai", {})
    xai_key = None
    if xai_entries:
        xai_key = st.selectbox("Explanation to include", ["None"] + list(xai_entries),
                               index=1 + list(xai_entries).index(st.session_state.get("xai_last", list(xai_entries)[0]))
                               if st.session_state.get("xai_last") in xai_entries else 0)
        xai_key = None if xai_key == "None" else xai_key
    else:
        st.caption("Tip: compute an explanation on the Results page to include it in the report.")

    if st.button("Create PDF report", type="primary"):
        with st.spinner("Building the report …"):
            wave = fig_to_png(plot_12lead(prep.signal_100[:LENGTH], FS, inp.leads,
                                          title=f"{inp.filename} — first 10 s at {FS} Hz"))
            xai = None
            if xai_key:
                e = xai_entries[xai_key]
                start = (e["window"] - 1) * LENGTH
                png = fig_to_png(plot_12lead(prep.signal_100[start:start + LENGTH], FS, inp.leads,
                                             attribution=e["attr"],
                                             title=f"{e['method']} — {LONG_NAMES[e['model']]} — {e['class']} "
                                                   f"(window {e['window']})"))
                xai = {"png": png, "model": e["model"], "class": e["class"], "method": e["method"]}
            data = {
                "report_id": st.session_state.report_id, "version": CFG.get("version", ""),
                "input": {"filename": inp.filename, "source": inp.source, "fs": inp.fs, "n_leads": len(inp.leads),
                          "duration_s": inp.duration_s, "ecg_id": inp.ecg_id, "true_labels": inp.true_labels,
                          "notes": prep.notes},
                "quality": [q.to_dict() for q in prep.quality], "quality_status": prep.quality_status,
                "heart_rate": prep.heart_rate, "classes": CLASSES, "model_names": NAMES,
                "predictions": res["models"], "performance": bundle.performance() if bundle else {},
                "waveform_png": wave, "xai": xai, "n_windows": res["n_windows"],
            }
            st.session_state.pdf = build_pdf(data)
    if st.session_state.get("pdf"):
        st.success("Report ready.")
        st.download_button("Download PDF report", st.session_state.pdf,
                           file_name=f"{st.session_state.report_id}.pdf", mime="application/pdf")

    st.markdown("## Other downloads")
    machine = {"report_id": st.session_state.report_id, "input": {"filename": inp.filename, "source": inp.source,
               "sampling_rate_hz": inp.fs, "duration_s": inp.duration_s, "ecg_id": inp.ecg_id,
               "reference_labels": inp.true_labels},
               "quality": [q.to_dict() for q in prep.quality], "heart_rate": prep.heart_rate,
               "results": res, "disclaimer": DISCLAIMER}
    c1, c2 = st.columns(2)
    c1.download_button("Download results (JSON)", json.dumps(machine, indent=2),
                       file_name=f"{st.session_state.report_id}.json", mime="application/json")
    buf = io.StringIO()
    pd.DataFrame(prep.signal_100, columns=inp.leads).assign(time_s=lambda d: d.index / FS).round(4).to_csv(buf, index=False)
    c2.download_button("Download processed signal (CSV, 100 Hz, mV)", buf.getvalue(),
                       file_name=f"{st.session_state.report_id}_signal_100hz.csv", mime="text/csv")


def page_about():
    st.markdown("# About & limitations")
    st.markdown("""
**What this is.** A university research prototype comparing three deep-learning architectures and two fusion
strategies for detecting five diagnostic superclasses in 12-lead ECGs, with explanations.

**Data.** PTB-XL v1.0.3 (PhysioNet): 21,799 clinical 12-lead ECGs from 18,869 patients, 10 s each.
The official stratified folds are used: folds 1–8 for training, fold 9 for validation (early stopping,
thresholds, ensemble weights, model selection) and fold 10 for the final test only. The folds keep each
patient in a single split.

**What it cannot do.**
- Identify specific conditions, their severity or urgency — only five broad superclasses.
- Guarantee performance on other devices, hospitals or populations. No external validation has been done.
- Give calibrated clinical risks. Probabilities are model scores.
- Explain clinical reasoning. Explanations show what influenced the model, which can differ from a cardiologist's criteria.

**Privacy.** Uploaded files are written to a temporary folder only while being parsed and are deleted
immediately. Nothing is stored after you close the page or press *Clear current ECG*.

**Please cite.** Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., Schaeffter, T. (2022). *PTB-XL, a large
publicly available electrocardiography dataset* (version 1.0.3). PhysioNet. — and Strodthoff, N. et al. (2021).
*Deep learning for ECG analysis: benchmarks and insights from PTB-XL.* IEEE JBHI.
""")
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


{"Overview": page_overview, "Analyze ECG": page_analyze, "Results": page_results,
 "Model comparison": page_comparison, "Report": page_report, "About & limitations": page_about}[page]()
