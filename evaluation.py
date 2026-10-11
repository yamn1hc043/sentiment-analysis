"""Standalone full-dataset evaluation for the Explainable Sentiment Analysis Framework."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, confusion_matrix

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL_MAP = {
    "LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE",
    "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL", "POSITIVE": "POSITIVE",
}
BATCH_SIZE = 32
POSITIVE_WORDS = {"love", "loved", "great", "excellent", "good", "amazing", "happy", "fantastic", "wonderful", "like", "best", "perfect", "awesome", "fast", "helpful", "satisfied", "impressive", "brilliant", "thrilled", "superb", "reliable"}
NEGATIVE_WORDS = {"hate", "hated", "bad", "terrible", "awful", "worst", "poor", "disappointed", "angry", "dislike", "slow", "broken", "late", "expensive", "problem", "annoying", "horrible", "failed", "useless", "damaged", "crash", "crashes", "lag", "laggy", "disastrous", "dissatisfied", "regret"}


def normalize_label(x):
    return LABEL_MAP.get(str(x).upper().strip(), str(x).upper().strip())


def conflict_level(text):
    words = re.findall(r"\b[\w']+\b", str(text).lower())
    pos = any(w in POSITIVE_WORDS for w in words)
    neg = any(w in NEGATIVE_WORDS for w in words)
    return "HIGH" if pos and neg else "LOW" if pos or neg else "NONE"


def load_data(path):
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="samples.csv")
    parser.add_argument("--output", default="evaluation_results.csv")
    parser.add_argument("--matrix", default="confusion_matrix.png")
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"Loaded {len(df):,} records from {args.input}")

    if "true_label" not in df.columns:
        raise ValueError("For evaluation, the dataset must contain a 'true_label' column.")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=None)

    texts = df["text"].fillna("").astype(str).tolist()
    predictions = []
    confidences = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        outputs = classifier(batch, batch_size=BATCH_SIZE, truncation=True)
        for output in outputs:
            if isinstance(output, dict):
                output = [output]
            scores = {label: 0.0 for label in LABELS}
            for item in output:
                label = normalize_label(item["label"])
                if label in scores:
                    scores[label] = float(item["score"])
            total = sum(scores.values())
            if total:
                scores = {k: v / total for k, v in scores.items()}
            label = max(scores, key=scores.get)
            predictions.append(label)
            confidences.append(scores[label])
        done = min(start + len(batch), len(texts))
        print(f"Processed {done:,} / {len(texts):,}")

    df["predicted_label"] = predictions
    df["confidence_score"] = np.round(confidences, 4)
    df["confidence_level"] = pd.cut(
        df["confidence_score"], bins=[-np.inf, .60, .80, np.inf], labels=["LOW", "MEDIUM", "HIGH"]
    ).astype(str)
    df["conflict_level"] = df["text"].apply(conflict_level)

    true_labels = df["true_label"].map(normalize_label)
    report = classification_report(true_labels, predictions, labels=LABELS, digits=4, zero_division=0)
    print("\n=== Classification Report ===")
    print(report)

    cm = confusion_matrix(true_labels, predictions, labels=LABELS)
    print("=== Confusion Matrix ===")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

    print("\n=== Framework-level diagnostics ===")
    print(f"Average confidence: {df['confidence_score'].mean():.2%}")
    print(f"Low-confidence predictions: {(df['confidence_level'] == 'LOW').sum():,}")
    print(f"High-conflict records: {(df['conflict_level'] == 'HIGH').sum():,}")

    df.to_csv(args.output, index=False)
    print(f"Saved complete results to {args.output}")

    fig, ax = plt.subplots(figsize=(6, 5))
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
    fig.savefig(args.matrix, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {args.matrix}")


if __name__ == "__main__":
    main()
