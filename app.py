import io
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Sentiment Analysis Framework",
    page_icon="🔎",
    layout="wide",
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
BATCH_SIZE = 32
MAX_LIME_FEATURES = 8
DEFAULT_LIME_SAMPLES = 200
MAX_REPRESENTATIVE_EXPLANATIONS = 18

LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "NEUTRAL": "NEUTRAL",
    "POSITIVE": "POSITIVE",
}

POSITIVE_WORDS = {
    "love", "loved", "lovely", "great", "excellent", "good", "amazing",
    "happy", "fantastic", "wonderful", "like", "liked", "best", "perfect",
    "awesome", "fast", "helpful", "satisfied", "impressive", "enjoyed",
    "brilliant", "thrilled", "superb", "reliable", "easy"
}

NEGATIVE_WORDS = {
    "hate", "hated", "bad", "terrible", "terribly", "awful", "worst", "poor",
    "disappointed", "disappointing", "angry", "dislike", "slow", "broken", "late",
    "expensive", "problem", "problems", "issue", "issues", "annoying", "horrible",
    "failed", "failure", "useless", "damaged", "defective", "crash", "crashes",
    "lag", "laggy", "disaster", "disastrous", "dissatisfied", "regret", "return"
}

STOP_WORDS = {
    "the", "is", "and", "in", "it", "this", "i", "with", "a", "of", "to",
    "was", "were", "for", "on", "that", "very", "my", "we", "but", "as",
    "an", "be", "are", "have", "has", "had", "from", "at", "or", "do",
    "does", "did", "not", "no"
}


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .hero { background: linear-gradient(135deg,#1f2937 0%,#111827 50%,#0f172a 100%);
            border:1px solid #30363d; border-radius:14px; padding:1.5rem 1.8rem; margin-bottom:1.5rem; }
    .hero-title { font-size:2rem; font-weight:700; color:#f3f4f6; margin-bottom:.4rem; }
    .hero-copy { color:#9ca3af; font-size:1rem; margin:0; }
    .section-card { background:#161b22; border:1px solid #30363d; border-radius:12px;
                    padding:1.2rem 1.5rem; margin-top:1rem; margin-bottom:1rem; }
    .section-card h3 { color:#f0f6fc !important; margin-top:0; margin-bottom:.4rem; }
    .section-card p { color:#8b949e !important; margin:0; }
    .stButton > button { border-radius:8px; font-weight:600; }
    .stTextArea textarea { background-color:#161b22 !important; color:#f0f6fc !important;
                           border:1px solid #30363d !important; border-radius:8px !important; }
    div[data-testid="stFileUploader"] { background-color:#161b22; border:1px dashed #30363d;
                                          border-radius:10px; padding:.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">🔎 Explainable Sentiment Analysis Framework</div>
      <p class="hero-copy">
        Full-dataset Transformer sentiment analysis with confidence, uncertainty,
        local LIME explanations, explanation-consistency checks, conflict detection,
        representative-case analysis, comparison and reporting.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading the Transformer model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
    )
    explainer = LimeTextExplainer(
        class_names=LABELS,
        random_state=42,
    )
    return sentiment_pipe, explainer


try:
    sentiment_pipeline, lime_explainer = load_model()
except Exception as exc:
    st.error("The Transformer model could not be loaded.")
    st.caption("Make sure the dependencies are installed and the model can be downloaded/cached.")
    st.exception(exc)
    st.stop()


# ============================================================
# NLP CORE
# ============================================================

def normalize_label(label: str) -> str:
    return LABEL_MAP.get(str(label).upper().strip(), str(label).upper().strip())


def probability_vector(texts: list[str]) -> np.ndarray:
    """Return [NEGATIVE, NEUTRAL, POSITIVE] probabilities for every text."""
    if isinstance(texts, str):
        texts = [texts]

    outputs = sentiment_pipeline(
        list(texts),
        batch_size=BATCH_SIZE,
        truncation=True,
    )

    rows = []
    for output in outputs:
        if isinstance(output, dict):
            output = [output]
        row = {label: 0.0 for label in LABELS}
        for item in output:
            label = normalize_label(item["label"])
            if label in row:
                row[label] = float(item["score"])
        total = sum(row.values())
        if total > 0:
            row = {k: v / total for k, v in row.items()}
        rows.append([row[label] for label in LABELS])
    return np.asarray(rows, dtype=float)


def predict_one(text: str) -> dict:
    probs = probability_vector([text])[0]
    idx = int(np.argmax(probs))
    label = LABELS[idx]
    confidence = float(probs[idx])

    if confidence >= 0.80:
        confidence_level = "HIGH"
    elif confidence >= 0.60:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return {
        "label": label,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(3)},
    }


def predict_many(texts: list[str]) -> list[dict]:
    probs = probability_vector(texts)
    results = []
    for row in probs:
        idx = int(np.argmax(row))
        label = LABELS[idx]
        confidence = float(row[idx])
        level = "HIGH" if confidence >= .80 else "MEDIUM" if confidence >= .60 else "LOW"
        results.append({
            "label": label,
            "confidence": confidence,
            "confidence_level": level,
            "probabilities": {LABELS[i]: float(row[i]) for i in range(3)},
        })
    return results


def lime_predict(texts: list[str]) -> np.ndarray:
    return probability_vector(list(texts))


def extract_keywords(text: str, num_keywords: int = 5) -> str:
    words = re.findall(r"\b[\w']+\b", str(text).lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ", ".join(w for w, _ in Counter(words).most_common(num_keywords))


def detect_sentiment_conflict(text: str) -> dict:
    words = re.findall(r"\b[\w']+\b", str(text).lower())
    pos = list(dict.fromkeys(w for w in words if w in POSITIVE_WORDS))
    neg = list(dict.fromkeys(w for w in words if w in NEGATIVE_WORDS))

    if pos and neg:
        level = "HIGH"
    elif pos or neg:
        level = "LOW"
    else:
        level = "NONE"

    return {
        "positive_evidence": ", ".join(pos) or "None detected",
        "negative_evidence": ", ".join(neg) or "None detected",
        "conflict_level": level,
        "positive_count": len(pos),
        "negative_count": len(neg),
    }


def explain_with_lime(text: str, num_samples: int = DEFAULT_LIME_SAMPLES) -> dict:
    explanation = lime_explainer.explain_instance(
        text,
        lime_predict,
        labels=(0, 1, 2),
        num_features=MAX_LIME_FEATURES,
        num_samples=num_samples,
    )
    prediction = predict_one(text)
    class_index = LABELS.index(prediction["label"])
    weights = explanation.as_list(label=class_index)
    return {"prediction": prediction, "weights": weights, "html": explanation.as_html()}


def calculate_explanation_reliability(
    text: str,
    weights: list[tuple[str, float]],
    original: dict,
    max_checks: int = 3,
) -> dict:
    """Validate LIME direction with controlled single-feature removal."""
    if not weights:
        return {"score": 0.0, "level": "UNAVAILABLE", "checks": []}

    predicted_label = original["label"]
    original_prob = original["probabilities"][predicted_label]
    checks = []
    valid = 0
    agreement = 0

    for feature, lime_weight in weights[:max_checks]:
        feature = str(feature).strip()
        if not feature or len(feature) < 2:
            continue
        pattern = re.compile(r"\b" + re.escape(feature) + r"\b", re.IGNORECASE)
        modified_text, replacements = pattern.subn("", text, count=1)
        if replacements == 0:
            continue
        modified_text = re.sub(r"\s+", " ", modified_text).strip()
        if not modified_text:
            continue

        modified = predict_one(modified_text)
        new_prob = modified["probabilities"][predicted_label]
        delta = new_prob - original_prob

        # For the predicted class: positive LIME weight should lose support
        # when the feature is removed; negative weight should gain support.
        agrees = delta < 0 if lime_weight > 0 else delta > 0 if lime_weight < 0 else False
        valid += 1
        agreement += int(agrees)
        checks.append({
            "token": feature,
            "lime_weight": round(float(lime_weight), 4),
            "original_probability": round(float(original_prob), 4),
            "modified_probability": round(float(new_prob), 4),
            "probability_change": round(float(delta), 4),
            "agrees": "YES" if agrees else "NO",
        })

    if valid == 0:
        return {"score": 0.0, "level": "UNAVAILABLE", "checks": checks}

    score = 100.0 * agreement / valid
    level = "HIGH" if score >= 75 else "MEDIUM" if score >= 50 else "LOW"
    return {"score": round(score, 1), "level": level, "checks": checks}


# ============================================================
# DATASET FUNCTIONS
# ============================================================

def read_uploaded_file(file):
    if file is None or getattr(file, "size", 0) == 0:
        st.error("Uploaded file is empty.")
        return None
    try:
        file.seek(0)
        name = file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(file, encoding_errors="ignore")
        if name.endswith(".xlsx"):
            return pd.read_excel(file)
        if name.endswith(".json"):
            return pd.read_json(file)
        if name.endswith(".txt"):
            raw = file.read()
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError:
                    raw = raw.decode("ISO-8859-1")
            return pd.DataFrame({"text": str(raw).splitlines()})
    except Exception as exc:
        st.error(f"Error reading file: {exc}")
    return None


def normalize_true_labels(series: pd.Series) -> pd.Series:
    return (
        series.fillna("").astype(str).str.upper().str.strip().replace({
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE",
        })
    )


def evaluate_dataframe(df: pd.DataFrame) -> dict:
    if "true_label" not in df.columns:
        return {}
    true = normalize_true_labels(df["true_label"])
    pred = df["label"]
    report = classification_report(
        true, pred, labels=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true, pred, labels=LABELS)
    return {"true": true, "report": report, "cm": cm}


def process_dataframe(df: pd.DataFrame, progress_placeholder=None) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("The uploaded file must contain a 'text' column.")

    result = df.copy()
    result["text"] = result["text"].fillna("").astype(str)
    texts = result["text"].tolist()
    predictions = []

    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        predictions.extend(predict_many(batch))
        if progress_placeholder is not None:
            done = min(start + len(batch), total)
            progress_placeholder.progress(
                done / max(total, 1),
                text=f"Analyzed {done:,} / {total:,} records",
            )

    result["label"] = [p["label"] for p in predictions]
    result["confidence"] = [round(p["confidence"], 4) for p in predictions]
    result["confidence_level"] = [p["confidence_level"] for p in predictions]
    result["keywords"] = result["text"].apply(extract_keywords)

    conflicts = result["text"].apply(detect_sentiment_conflict)
    result["positive_evidence"] = conflicts.apply(lambda x: x["positive_evidence"])
    result["negative_evidence"] = conflicts.apply(lambda x: x["negative_evidence"])
    result["conflict_level"] = conflicts.apply(lambda x: x["conflict_level"])
    result["explanation_status"] = "Representative-case LIME available"

    return result


def dataset_summary(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    conflict_counts = df["conflict_level"].value_counts().reindex(["HIGH", "LOW", "NONE"], fill_value=0)
    uncertainty_counts = df["confidence_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    evaluation = evaluate_dataframe(df)

    avg_reliability = None
    if "explanation_reliability" in df.columns:
        numeric = pd.to_numeric(df["explanation_reliability"], errors="coerce").dropna()
        if len(numeric):
            avg_reliability = float(numeric.mean())

    return {
        "counts": counts,
        "conflict_counts": conflict_counts,
        "uncertainty_counts": uncertainty_counts,
        "average_confidence": float(df["confidence"].mean()) if len(df) else 0.0,
        "low_confidence": int((df["confidence_level"] == "LOW").sum()),
        "high_conflict": int((df["conflict_level"] == "HIGH").sum()),
        "avg_explanation_reliability": avg_reliability,
        "evaluation": evaluation,
    }


def top_evidence_terms(df: pd.DataFrame, column: str, top_n: int = 10):
    counter = Counter()
    for value in df[column].fillna(""):
        for word in re.findall(r"\b[\w']+\b", str(value).lower()):
            if word not in {"none", "detected"}:
                counter[word] += 1
    return counter.most_common(top_n)


def select_representative_cases(df: pd.DataFrame, max_cases: int = MAX_REPRESENTATIVE_EXPLANATIONS) -> pd.DataFrame:
    """Select informative rows rather than running LIME on every record."""
    if df.empty:
        return df.copy()

    selected_indices = []

    # Lowest-confidence cases are useful for uncertainty analysis.
    selected_indices.extend(df.nsmallest(min(4, len(df)), "confidence").index.tolist())

    # High-conflict cases are useful for ambiguity analysis.
    high_conflict = df[df["conflict_level"] == "HIGH"]
    if not high_conflict.empty:
        selected_indices.extend(high_conflict.nsmallest(4, "confidence").index.tolist())

    # At least a few representative examples from each predicted class.
    for label in LABELS:
        subset = df[df["label"] == label]
        if not subset.empty:
            selected_indices.extend(subset.nlargest(2, "confidence").index.tolist())

    # Deduplicate while preserving order.
    selected_indices = list(dict.fromkeys(selected_indices))[:max_cases]
    return df.loc[selected_indices].copy()


def run_representative_explanations(df: pd.DataFrame, progress_placeholder=None) -> pd.DataFrame:
    cases = select_representative_cases(df)
    if cases.empty:
        return cases

    rows = []
    total = len(cases)
    for i, (idx, row) in enumerate(cases.iterrows(), start=1):
        text = row["text"]
        try:
            explanation = explain_with_lime(text, DEFAULT_LIME_SAMPLES)
            reliability = calculate_explanation_reliability(
                text, explanation["weights"], explanation["prediction"]
            )
            top_drivers = "; ".join(
                f"{token}: {weight:+.3f}" for token, weight in explanation["weights"][:5]
            )
            rows.append({
                "row_index": idx,
                "text": text,
                "label": row["label"],
                "confidence": row["confidence"],
                "confidence_level": row["confidence_level"],
                "conflict_level": row["conflict_level"],
                "explanation_reliability": reliability["score"] if reliability["level"] != "UNAVAILABLE" else np.nan,
                "reliability_level": reliability["level"],
                "top_lime_drivers": top_drivers,
            })
        except Exception as exc:
            rows.append({
                "row_index": idx,
                "text": text,
                "label": row["label"],
                "confidence": row["confidence"],
                "confidence_level": row["confidence_level"],
                "conflict_level": row["conflict_level"],
                "explanation_reliability": np.nan,
                "reliability_level": "ERROR",
                "top_lime_drivers": str(exc),
            })
        if progress_placeholder is not None:
            progress_placeholder.progress(i / total, text=f"LIME: {i} / {total} representative cases")

    return pd.DataFrame(rows)


# ============================================================
# CHARTS / PDF
# ============================================================

def make_distribution_chart(counts):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=.2)
    fig.tight_layout()
    return fig


def make_conflict_chart(counts):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Sentiment Conflict Evidence")
    ax.set_xlabel("Conflict Level")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=.2)
    fig.tight_layout()
    return fig


def make_confusion_chart(cm):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(3), ["POS", "NEG", "NEU"])
    ax.set_yticks(range(3), ["POS", "NEG", "NEU"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def fig_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def create_comparison_pdf(primary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> bytes:
    """Create a side-by-side A/B analytical PDF using model predictions only."""
    a = dataset_summary(primary_df)
    b = dataset_summary(comparison_df)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    elements = [Paragraph("A/B Sentiment & Explainability Comparison Report", styles["Title"]), Spacer(1, 10)]

    n_a, n_b = len(primary_df), len(comparison_df)
    pct = lambda x, n: 100 * x / max(n, 1)
    rows = [
        ["Metric", "Primary Dataset", "Comparison Dataset", "Change (B - A)"],
        ["Records", f"{n_a:,}", f"{n_b:,}", f"{n_b-n_a:+,}"],
        ["Positive %", f"{pct(a['counts']['POSITIVE'], n_a):.1f}%", f"{pct(b['counts']['POSITIVE'], n_b):.1f}%", f"{pct(b['counts']['POSITIVE'], n_b)-pct(a['counts']['POSITIVE'], n_a):+.1f} pp"],
        ["Negative %", f"{pct(a['counts']['NEGATIVE'], n_a):.1f}%", f"{pct(b['counts']['NEGATIVE'], n_b):.1f}%", f"{pct(b['counts']['NEGATIVE'], n_b)-pct(a['counts']['NEGATIVE'], n_a):+.1f} pp"],
        ["Neutral %", f"{pct(a['counts']['NEUTRAL'], n_a):.1f}%", f"{pct(b['counts']['NEUTRAL'], n_b):.1f}%", f"{pct(b['counts']['NEUTRAL'], n_b)-pct(a['counts']['NEUTRAL'], n_a):+.1f} pp"],
        ["Average confidence", f"{a['average_confidence']:.1%}", f"{b['average_confidence']:.1%}", f"{b['average_confidence']-a['average_confidence']:+.1%}"],
        ["Low-confidence %", f"{pct(a['low_confidence'], n_a):.1f}%", f"{pct(b['low_confidence'], n_b):.1f}%", f"{pct(b['low_confidence'], n_b)-pct(a['low_confidence'], n_a):+.1f} pp"],
        ["High-conflict %", f"{pct(a['high_conflict'], n_a):.1f}%", f"{pct(b['high_conflict'], n_b):.1f}%", f"{pct(b['high_conflict'], n_b)-pct(a['high_conflict'], n_a):+.1f} pp"],
    ]
    table = Table(rows, repeatRows=1, colWidths=[125, 120, 135, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .5, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Combined sentiment distribution chart.
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width/2, [a["counts"][k] for k in labels], width, label="Primary")
    ax.bar(x + width/2, [b["counts"][k] for k in labels], width, label="Comparison")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Records")
    ax.set_title("Sentiment Distribution Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=.2)
    fig.tight_layout()
    elements.append(Paragraph("Sentiment Distribution Comparison", styles["Heading2"]))
    elements.append(Image(fig_bytes(fig), width=420, height=240))
    elements.append(Spacer(1, 8))

    # Confidence/conflict comparison chart.
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    metrics = ["Avg confidence", "Low-confidence", "High-conflict"]
    av = [a["average_confidence"] * 100, pct(a["low_confidence"], n_a), pct(a["high_conflict"], n_a)]
    bv = [b["average_confidence"] * 100, pct(b["low_confidence"], n_b), pct(b["high_conflict"], n_b)]
    ax2.bar(x - width/2, av, width, label="Primary")
    ax2.bar(x + width/2, bv, width, label="Comparison")
    ax2.set_xticks(x, metrics)
    ax2.set_ylabel("Percent")
    ax2.set_title("Confidence and Conflict Comparison")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=.2)
    fig2.tight_layout()
    elements.append(Paragraph("Confidence and Conflict Comparison", styles["Heading2"]))
    elements.append(Image(fig_bytes(fig2), width=420, height=240))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "Interpretation: the report compares model-generated sentiment distributions, confidence and conflict statistics. "
        "Ground-truth accuracy is not calculated unless labels are available; this A/B report is intended for unlabeled dataset comparison.",
        styles["BodyText"],
    ))
    doc.build(elements)
    return buffer.getvalue()


def create_pdf_report(df: pd.DataFrame, title: str, representative: pd.DataFrame | None = None) -> bytes:
    summary = dataset_summary(df)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    elements = [Paragraph(title, styles["Title"]), Spacer(1, 10)]

    overview = [
        ["Metric", "Value"],
        ["Records analyzed", f"{len(df):,}"],
        ["Average confidence", f"{summary['average_confidence']:.2%}"],
        ["Low-confidence predictions", f"{summary['low_confidence']:,}"],
        ["High-conflict records", f"{summary['high_conflict']:,}"],
        ["Positive / Negative / Neutral", f"{summary['counts']['POSITIVE']} / {summary['counts']['NEGATIVE']} / {summary['counts']['NEUTRAL']}"],
    ]
    table = Table(overview, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .5, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Sentiment Distribution", styles["Heading2"]))
    elements.append(Image(fig_bytes(make_distribution_chart(summary["counts"])), width=420, height=240))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Conflict Analysis", styles["Heading2"]))
    elements.append(Image(fig_bytes(make_conflict_chart(summary["conflict_counts"])), width=420, height=240))
    elements.append(Spacer(1, 8))

    evaluation = summary["evaluation"]
    if evaluation:
        elements.append(Paragraph("Ground-Truth Evaluation", styles["Heading2"]))
        report = evaluation["report"]
        rows = [["Class", "Precision", "Recall", "F1", "Support"]]
        for label in LABELS:
            r = report[label]
            rows.append([label, f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1-score']:.3f}", int(r['support'])])
        rows.append(["Accuracy", "", "", f"{report['accuracy']:.3f}", int(sum(report[x]['support'] for x in LABELS))])
        et = Table(rows, repeatRows=1)
        et.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), .5, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        elements.append(et)
        elements.append(Spacer(1, 8))
        elements.append(Image(fig_bytes(make_confusion_chart(evaluation["cm"])), width=360, height=290))

    pos = top_evidence_terms(df, "positive_evidence")
    neg = top_evidence_terms(df, "negative_evidence")
    elements.append(Paragraph("Frequent Evidence Terms", styles["Heading2"]))
    elements.append(Paragraph("Positive: " + (", ".join(f"{w} ({n})" for w,n in pos) or "None"), styles["BodyText"]))
    elements.append(Paragraph("Negative: " + (", ".join(f"{w} ({n})" for w,n in neg) or "None"), styles["BodyText"]))

    if representative is not None and not representative.empty:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Representative Explainability Cases", styles["Heading2"]))
        rows = [["Label", "Confidence", "Conflict", "Reliability", "Top LIME drivers"]]
        for _, r in representative.iterrows():
            rows.append([
                str(r["label"]),
                f"{float(r['confidence']):.2%}",
                str(r["conflict_level"]),
                "N/A" if pd.isna(r["explanation_reliability"]) else f"{float(r['explanation_reliability']):.0f}%",
                str(r["top_lime_drivers"])[:150],
            ])
        rt = Table(rows, repeatRows=1, colWidths=[60,65,55,60,250])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), .5, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        elements.append(rt)

    doc.build(elements)
    return buffer.getvalue()


# ============================================================
# DATASET DASHBOARD
# ============================================================

def render_dataset_dashboard(df: pd.DataFrame, key_prefix: str):
    summary = dataset_summary(df)

    st.subheader("Dataset Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records analyzed", f"{len(df):,}")
    c2.metric("Avg confidence", f"{summary['average_confidence']:.1%}")
    c3.metric("Low confidence", f"{summary['low_confidence']:,}")
    c4.metric("High conflict", f"{summary['high_conflict']:,}")
    c5.metric("Dominant sentiment", summary["counts"].idxmax())

    if summary["evaluation"]:
        accuracy = summary["evaluation"]["report"]["accuracy"]
        st.success(f"Ground-truth evaluation available • Accuracy: {accuracy:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(make_distribution_chart(summary["counts"]), clear_figure=True)
    with col2:
        st.pyplot(make_conflict_chart(summary["conflict_counts"]), clear_figure=True)

    st.subheader("Complete Results")
    st.caption(f"All {len(df):,} uploaded records were analyzed. The table is scrollable; exports contain the complete dataset.")
    st.dataframe(df, use_container_width=True, height=450)

    st.subheader("Frequent Evidence")
    e1, e2 = st.columns(2)
    pos_terms = top_evidence_terms(df, "positive_evidence")
    neg_terms = top_evidence_terms(df, "negative_evidence")
    e1.write("**Positive evidence:** " + (", ".join(f"{w} ({n})" for w,n in pos_terms) or "None"))
    e2.write("**Negative evidence:** " + (", ".join(f"{w} ({n})" for w,n in neg_terms) or "None"))

    st.subheader("Representative Explainability")
    st.caption("Full-dataset prediction is performed on every row. LIME is reserved for informative cases so large datasets remain practical.")
    if st.button("Run LIME on Representative Cases", key=f"lime_{key_prefix}"):
        progress = st.progress(0, text="Selecting representative cases...")
        rep = run_representative_explanations(df, progress)
        progress.empty()
        st.session_state[f"rep_{key_prefix}"] = rep

    if f"rep_{key_prefix}" in st.session_state:
        rep = st.session_state[f"rep_{key_prefix}"]
        st.dataframe(rep, use_container_width=True, height=320)
        numeric = pd.to_numeric(rep["explanation_reliability"], errors="coerce").dropna()
        if len(numeric):
            st.metric("Average representative explanation reliability", f"{numeric.mean():.1f}%")

    st.subheader("Export")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("Download complete CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{key_prefix}_complete_results.csv", mime="text/csv", use_container_width=True)
    with e2:
        st.download_button("Download complete JSON", df.to_json(orient="records", indent=2).encode("utf-8"), file_name=f"{key_prefix}_complete_results.json", mime="application/json", use_container_width=True)
    with e3:
        pdf = create_pdf_report(df, f"{key_prefix.title()} Sentiment & Explainability Report", st.session_state.get(f"rep_{key_prefix}"))
        st.download_button("Download analytical PDF", pdf, file_name=f"{key_prefix}_analytical_report.pdf", mime="application/pdf", use_container_width=True)


# ============================================================
# MODULE 1 — PRIMARY DATASET
# ============================================================

st.markdown(
    """
    <div class="section-card">
      <h3>1. Full-Dataset Analysis</h3>
      <p>
        Upload CSV, Excel, JSON or TXT data. The <b>entire uploaded dataset</b>
        is analyzed. A <b>true_label</b> column is optional but recommended for evaluation.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Choose primary dataset", type=["csv", "xlsx", "json", "txt"], key="primary_file")

if uploaded_file is not None:
    raw_df = read_uploaded_file(uploaded_file)
    if raw_df is not None:
        if "text" not in raw_df.columns:
            st.warning("The uploaded file must contain a 'text' column.")
        else:
            label_message = "Ground truth detected: true_label" if "true_label" in raw_df.columns else "No true_label column — evaluation will be skipped."
            st.info(f"Loaded {len(raw_df):,} rows • {label_message}")
            if st.button("Analyze ALL Records", type="primary"):
                progress = st.progress(0, text="Starting full-dataset analysis...")
                try:
                    processed = process_dataframe(raw_df, progress)
                    progress.empty()
                    st.session_state["primary_processed_df"] = processed
                    st.session_state.pop("rep_primary", None)
                    st.success(f"Completed analysis of all {len(processed):,} records.")
                except Exception as exc:
                    progress.empty()
                    st.error(f"Dataset analysis failed: {exc}")
                    st.exception(exc)

if "primary_processed_df" in st.session_state:
    render_dataset_dashboard(st.session_state["primary_processed_df"], "primary")


# ============================================================
# MODULE 2 — A/B COMPARISON
# ============================================================

st.markdown(
    """
    <div class="section-card">
      <h3>2. Explainability-Oriented A/B Comparison</h3>
      <p>Compare sentiment, confidence, uncertainty and conflict across two complete datasets.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

comparison_file = st.file_uploader("Choose comparison dataset", type=["csv", "xlsx", "json", "txt"], key="comparison_file")

if comparison_file is not None:
    comp_raw = read_uploaded_file(comparison_file)
    if comp_raw is not None:
        if "text" not in comp_raw.columns:
            st.warning("Comparison file must contain a 'text' column.")
        elif st.button("Analyze ALL Comparison Records"):
            progress = st.progress(0, text="Analyzing comparison dataset...")
            try:
                st.session_state["comparison_processed_df"] = process_dataframe(comp_raw, progress)
                st.session_state.pop("rep_comparison", None)
                progress.empty()
                st.success(f"Completed comparison analysis of all {len(comp_raw):,} records.")
            except Exception as exc:
                progress.empty()
                st.error(f"Comparison analysis failed: {exc}")
                st.exception(exc)

if "primary_processed_df" in st.session_state and "comparison_processed_df" in st.session_state:
    a = dataset_summary(st.session_state["primary_processed_df"])
    b = dataset_summary(st.session_state["comparison_processed_df"])

    def pct(x, total):
        return f"{100*x/max(total,1):.1f}%"

    n_a = len(st.session_state["primary_processed_df"])
    n_b = len(st.session_state["comparison_processed_df"])
    comparison_table = pd.DataFrame({
        "Metric": [
            "Records", "Positive %", "Negative %", "Neutral %",
            "Average confidence", "Low-confidence %", "High-conflict %",
        ],
        "Primary Dataset": [
            n_a,
            pct(a["counts"]["POSITIVE"], n_a),
            pct(a["counts"]["NEGATIVE"], n_a),
            pct(a["counts"]["NEUTRAL"], n_a),
            f"{a['average_confidence']:.1%}",
            pct(a["low_confidence"], n_a),
            pct(a["high_conflict"], n_a),
        ],
        "Comparison Dataset": [
            n_b,
            pct(b["counts"]["POSITIVE"], n_b),
            pct(b["counts"]["NEGATIVE"], n_b),
            pct(b["counts"]["NEUTRAL"], n_b),
            f"{b['average_confidence']:.1%}",
            pct(b["low_confidence"], n_b),
            pct(b["high_conflict"], n_b),
        ],
    })

    st.subheader("A/B Explainability Summary")
    st.dataframe(comparison_table, use_container_width=True)

    chart_data = pd.DataFrame({"Primary": a["counts"], "Comparison": b["counts"]})
    st.bar_chart(chart_data)

    st.subheader("Comparison Report")
    st.download_button(
        "Download A/B Analytical PDF",
        create_comparison_pdf(
            st.session_state["primary_processed_df"],
            st.session_state["comparison_processed_df"],
        ),
        file_name="ab_sentiment_comparison_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ============================================================
# MODULE 3 — INDIVIDUAL EXPLAINABILITY
# ============================================================

st.markdown(
    """
    <div class="section-card">
      <h3>3. Individual Explainability & Reliability</h3>
      <p>Inspect model probabilities, LIME attribution, prediction uncertainty and explanation consistency.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_area("Enter text", placeholder="Example: The product quality was excellent, but the delivery was terribly slow.", height=130)
lime_samples = st.slider("LIME sampling budget", 100, 500, DEFAULT_LIME_SAMPLES, 50)

if st.button("Analyze Individual Text", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running Transformer + LIME + reliability analysis..."):
            try:
                prediction = predict_one(user_input)
                conflict = detect_sentiment_conflict(user_input)
                explanation = explain_with_lime(user_input, lime_samples)
                reliability = calculate_explanation_reliability(user_input, explanation["weights"], prediction)
                st.session_state["last_analysis"] = {
                    "prediction": prediction,
                    "conflict": conflict,
                    "explanation": explanation,
                    "reliability": reliability,
                }
            except Exception as exc:
                st.error(f"Individual analysis failed: {exc}")
                st.exception(exc)

if "last_analysis" in st.session_state:
    result = st.session_state["last_analysis"]
    prediction = result["prediction"]
    conflict = result["conflict"]
    explanation = result["explanation"]
    reliability = result["reliability"]

    st.subheader("Prediction & Reliability")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment", prediction["label"])
    c2.metric("Confidence", f"{prediction['confidence']:.2%}")
    c3.metric("Prediction confidence", prediction["confidence_level"])
    c4.metric("Explanation reliability", "N/A" if reliability["level"] == "UNAVAILABLE" else f"{reliability['score']:.0f}%")

    if prediction["confidence_level"] == "LOW":
        st.warning("Low-confidence prediction: the model has limited separation between sentiment classes.")
    elif prediction["confidence_level"] == "MEDIUM":
        st.info("Medium-confidence prediction: the result should be interpreted with some caution.")

    st.subheader("Class Probabilities")
    probability_df = pd.DataFrame({"Sentiment": LABELS, "Probability": [prediction["probabilities"][x] * 100 for x in LABELS]})
    st.bar_chart(probability_df.set_index("Sentiment")["Probability"])

    st.subheader(f"LIME Explanation — influence on predicted {prediction['label']} class")
    styled_html = f'<div style="background:#fff;color:#000;padding:15px;border-radius:10px;font-family:Arial">{explanation["html"]}</div>'
    components.html(styled_html, height=480, scrolling=True)

    weights_df = pd.DataFrame(explanation["weights"], columns=["Token", "LIME Weight"])
    weights_df["Effect on predicted class"] = weights_df["LIME Weight"].apply(lambda x: "Supports predicted class" if x > 0 else "Opposes predicted class")
    weights_df["LIME Weight"] = weights_df["LIME Weight"].round(4)
    st.dataframe(weights_df, use_container_width=True)

    st.subheader("Explanation Reliability Check")
    if reliability["level"] == "UNAVAILABLE":
        st.info("Reliability could not be calculated for the selected features.")
    else:
        r1, r2 = st.columns(2)
        r1.metric("Consistency score", f"{reliability['score']:.1f}%")
        r2.metric("Reliability level", reliability["level"])
        st.caption("Consistency score = agreement between LIME's direction and the model's probability change after controlled feature removal. It is not a calibrated probability.")
        if reliability["checks"]:
            st.dataframe(pd.DataFrame(reliability["checks"]), use_container_width=True)

    st.subheader("Sentiment Conflict Analysis")
    q1, q2, q3 = st.columns(3)
    q1.metric("Conflict level", conflict["conflict_level"])
    q2.write("**Positive evidence**")
    q2.write(conflict["positive_evidence"])
    q3.write("**Negative evidence**")
    q3.write(conflict["negative_evidence"])
    st.caption("Conflict detection is a secondary lexical signal; the Transformer remains the primary sentiment classifier.")


# ============================================================
# MODULE 4 — EVALUATION
# ============================================================

st.markdown(
    """
    <div class="section-card">
      <h3>4. Ground-Truth Evaluation</h3>
      <p>If <b>true_label</b> is present, the complete uploaded dataset is evaluated against ground truth.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "primary_processed_df" in st.session_state:
    eval_result = evaluate_dataframe(st.session_state["primary_processed_df"])
    if eval_result:
        report_df = pd.DataFrame(eval_result["report"]).transpose()
        st.dataframe(report_df, use_container_width=True)
        st.pyplot(make_confusion_chart(eval_result["cm"]), clear_figure=True)

st.caption("Full-dataset prediction is performed for every uploaded row. LIME is intentionally limited to representative/ambiguous cases to keep large datasets practical.")
