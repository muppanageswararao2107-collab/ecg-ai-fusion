"""PDF report generation (reportlab). Only values that were actually computed are printed;
anything else is shown as "not available"."""
from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

DISCLAIMER = ("Research prototype for educational purposes. Not a medical device and not a substitute for "
              "evaluation by a qualified clinician.")
TEAL = colors.HexColor("#0f766e")
STATUS_COLORS = {"PASS": colors.HexColor("#15803d"), "WARN": colors.HexColor("#b45309"),
                 "FAIL": colors.HexColor("#b91c1c")}


def _fmt(v, nd=3):
    if v is None:
        return "not available"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "not available" if f != f else f"{f:.{nd}f}"


def _table(rows, col_widths, header=True, zebra=True):
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d4d4d8")),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5)]
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f4f4f5")))
    t.setStyle(TableStyle(style))
    return t


def build_pdf(data: dict) -> bytes:
    """data keys:
      report_id, input{filename, source, fs, n_leads, duration_s, ecg_id, true_labels, notes[]},
      quality[{name,status,message}], quality_status, heart_rate{available,bpm,reason},
      classes[], model_names{key:display}, predictions{model:{probs, flags, thresholds, branch_weights?, weights?}},
      performance{model:{test_macro_auroc,...}} or {}, waveform_png bytes, xai{png, model, class, method} or None,
      n_windows, version
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm,
                            title=f"ECG analysis report {data['report_id']}", author="ECG AI research prototype")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, textColor=TEAL, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=TEAL, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=8.8, leading=11.5, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10, textColor=colors.HexColor("#52525b"))
    warn = ParagraphStyle("warn", parent=body, backColor=colors.HexColor("#fef3c7"), borderPadding=5,
                          textColor=colors.HexColor("#78350f"))
    W = doc.width
    story = []

    inp = data["input"]
    story.append(Paragraph("ECG Analysis Report", h1))
    story.append(Paragraph(
        f"Report ID <b>{data['report_id']}</b> &nbsp;·&nbsp; Generated "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp; Software v{data.get('version', '')}", small))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>{DISCLAIMER}</b>", warn))

    story.append(Paragraph("1. Recording", h2))
    rows = [["Field", "Value"],
            ["Input source", inp.get("source", "")],
            ["File / record", str(inp.get("filename", ""))],
            ["Original sampling rate", f"{inp['fs']:g} Hz" if inp.get("fs") else "not available"],
            ["Leads", str(inp.get("n_leads", "not available"))],
            ["Duration", f"{inp['duration_s']:.1f} s" if inp.get("duration_s") else "not available"],
            ["10 s windows analysed", str(data.get("n_windows", "not available"))]]
    if inp.get("ecg_id") is not None:
        rows.append(["PTB-XL ecg_id", str(inp["ecg_id"])])
    if inp.get("true_labels") is not None:
        rows.append(["PTB-XL reference labels", ", ".join(inp["true_labels"]) or "none"])
    story.append(_table(rows, [W * 0.32, W * 0.68]))
    if inp.get("notes"):
        story.append(Spacer(1, 3))
        story.append(Paragraph("Processing notes: " + " ".join(inp["notes"]), small))

    story.append(Paragraph("2. Signal quality", h2))
    qrows = [["Check", "Status", "Details"]] + [[q["name"], q["status"], Paragraph(q["message"], body)]
                                                for q in data["quality"]]
    qt = _table(qrows, [W * 0.25, W * 0.1, W * 0.65])
    for r, q in enumerate(data["quality"], start=1):
        qt.setStyle(TableStyle([("TEXTCOLOR", (1, r), (1, r), STATUS_COLORS.get(q["status"], colors.black)),
                                ("FONT", (1, r), (1, r), "Helvetica-Bold", 8.5)]))
    story.append(qt)
    hr = data.get("heart_rate") or {}
    hr_txt = (f"Heart rate (automated estimate, lead {hr.get('lead', 'II')}): <b>{hr['bpm']:.0f} bpm</b>. "
              "Not a validated clinical measurement." if hr.get("available")
              else f"Heart rate: not available ({hr.get('reason', 'not computed')}).")
    story.append(Spacer(1, 3))
    story.append(Paragraph(hr_txt, body))
    story.append(Paragraph("No other measurements (intervals, axis) are computed by this system.", small))

    story.append(Paragraph("3. ECG waveform", h2))
    story.append(KeepTogether([
        Image(io.BytesIO(data["waveform_png"]), width=W * 0.92, height=W * 0.92 * 0.6),
        Paragraph("First 10 s at 100 Hz (the signal the models analysed), standard paper grid "
                  "0.2 s × 0.5 mV.", small)]))

    story.append(Paragraph("4. Model outputs", h2))
    classes = data["classes"]
    names = data["model_names"]
    preds = data["predictions"]
    order = [m for m in ["cnn", "lstm", "transformer", "ensemble", "fusion"] if m in preds]
    head = ParagraphStyle("head", parent=body, textColor=colors.white, fontName="Helvetica-Bold", fontSize=8.2,
                          leading=9.5)
    prow = [["Class"] + [Paragraph(names.get(m, m), head) for m in order]]
    for c in classes:
        prow.append([c] + [f"{preds[m]['probs'][c]:.2f}{' *' if preds[m]['flags'][c] else ''}" for m in order])
    story.append(_table(prow, [W * 0.12] + [W * 0.88 / len(order)] * len(order)))
    story.append(Paragraph(
        "Values are model probabilities (0–1) averaged across 10 s windows. * = above that model's decision "
        "threshold, which was tuned on the validation fold (PTB-XL fold 9) only. Classes: NORM normal ECG, "
        "MI myocardial infarction, STTC ST/T change, CD conduction disturbance, HYP hypertrophy. "
        "These are model outputs, not diagnoses.", small))
    if "fusion" in preds and preds["fusion"].get("branch_weights"):
        bw = preds["fusion"]["branch_weights"]
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Parallel-fusion branch weights for this ECG (learned gate): " +
            ", ".join(f"{k.upper()} {v * 100:.0f}%" for k, v in bw.items()) +
            ". These show how the fusion model weighted each branch; they are not a measure of correctness.", body))
    if "ensemble" in preds and preds["ensemble"].get("weights"):
        story.append(Paragraph("Late-fusion ensemble weights (tuned on validation): " +
                               ", ".join(f"{k.upper()} {v:.1f}" for k, v in preds["ensemble"]["weights"].items()) + ".",
                               body))

    story.append(Paragraph("5. Model performance on the held-out PTB-XL test fold", h2))
    perf = data.get("performance") or {}
    if perf:
        rows = [["Model", "Test macro-AUROC", "Validation macro-AUROC", "Parameters"]]
        for m in order:
            p = perf.get(m, {})
            rows.append([names.get(m, m), _fmt(p.get("test_macro_auroc")), _fmt(p.get("val_macro_auroc")),
                         f"{p['n_params']:,}" if p.get("n_params") else "—"])
        story.append(_table(rows, [W * 0.4, W * 0.2, W * 0.22, W * 0.18]))
        story.append(Paragraph("Read from the saved evaluation results (PTB-XL fold 10). Performance on recordings "
                               "from other devices or populations may be lower.", small))
    else:
        story.append(Paragraph("Evaluation results not available.", body))

    if data.get("xai"):
        x = data["xai"]
        block = [Paragraph("6. Explanation", h2),
                 Paragraph(f"{x['method']} for model <b>{names.get(x['model'], x['model'])}</b>, class "
                           f"<b>{x['class']}</b>. Red shading marks the time regions that most influenced this "
                           "output. Highlighted regions are model behaviour, not clinical findings.", body),
                 Spacer(1, 3), Image(io.BytesIO(x["png"]), width=W, height=W * 0.6)]
        story.append(KeepTogether(block))

    lim = [Paragraph("Limitations", h2)]
    for line in [
        "Trained and evaluated on one public dataset (PTB-XL, recordings from 1989–1996, one device manufacturer).",
        "Detects five broad diagnostic superclasses only; it cannot identify specific conditions, severity or urgency.",
        "Uploads from other devices, populations or with different filtering may be out of distribution.",
        "No external validation has been performed. Probabilities are not calibrated clinical risks.",
        "Explanation maps show what influenced the model, which may not match clinical reasoning.",
    ]:
        lim.append(Paragraph("• " + line, body))
    lim += [Spacer(1, 6), Paragraph(f"<b>{DISCLAIMER}</b>", warn)]
    story.append(KeepTogether(lim))

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#71717a"))
        canvas.drawString(15 * mm, 9 * mm, f"Report {data['report_id']} · research prototype · not for clinical use")
        canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
