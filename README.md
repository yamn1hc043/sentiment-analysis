# Explainable Sentiment Analysis Framework

A course-project framework for **three-class sentiment analysis with model explainability, uncertainty analysis, conflict detection, and dataset-level reporting**.

## Main contribution

The project does not claim a new Transformer or a new LIME algorithm. Its contribution is a unified workflow that combines:

1. **Three-class Transformer sentiment classification** — NEGATIVE, NEUTRAL, POSITIVE.
2. **Prediction confidence and uncertainty levels** — HIGH / MEDIUM / LOW.
3. **Local LIME explanations** — word-level attribution for individual and representative cases.
4. **Explanation consistency checking** — controlled feature removal tests whether the model response agrees with LIME's direction.
5. **Sentiment conflict detection** — identifies text containing both positive and negative evidence.
6. **Full-dataset analytics** — every uploaded record receives a prediction, confidence and conflict signal.
7. **Representative-case XAI** — LIME is run on informative cases rather than thousands of rows, keeping large datasets practical.
8. **A/B dataset comparison** — compares sentiment, confidence, uncertainty and conflict statistics.
9. **Ground-truth evaluation** — accuracy, precision, recall, F1 and confusion matrix when `true_label` exists.
10. **Analytical reports** — complete CSV/JSON exports plus a summary PDF.

## Input format

CSV / XLSX / JSON / TXT files are supported.

For CSV/XLSX/JSON, the required column is:

```text
text
```

Recommended evaluation format:

```text
text,true_label
"I love this product",POSITIVE
"This is terrible",NEGATIVE
"The package arrived today",NEUTRAL
```

`true_label` is optional for prediction, but required for ground-truth evaluation.

## Datasets in this project

- `demo.csv` — 10 labeled examples for a quick UI smoke test.
- `comparison_dataset.csv` — 10 labeled examples with a different sentiment distribution for A/B comparison.
- `samples.csv` — 50 balanced labeled examples (17 POSITIVE, 17 NEGATIVE, 16 NEUTRAL) for standalone evaluation.

For the final review, a larger real labeled dataset is preferable to the small demo files. The application itself is designed to analyze **all uploaded records**.

## Architecture

```text
                 Input Dataset / Text
                         |
                         v
              Transformer Sentiment Model
                         |
            +------------+------------+
            |            |            |
        Prediction   Confidence    Probabilities
            |            |
            |       Uncertainty Level
            |
            +-------> Conflict Signal
            |
            v
          LIME
            |
            v
   Explanation Consistency
            |
            v
   Explainability Dashboard
            |
     +------+-------+-------+
     |              |       |
  Full Dataset    A/B    Reports
   Analytics    Compare   /Export
```

## Important scalability design

The framework predicts **every record** in batches.

It does **not** run LIME on every record by default. LIME creates many perturbed samples per text, so applying it to thousands of rows would be unnecessarily slow. Instead, the app selects representative cases such as:

- lowest-confidence predictions,
- high-conflict records,
- representative examples from each predicted class.

This gives full-dataset sentiment analysis while retaining detailed local explainability where it is most informative.

## Run the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run standalone evaluation

The default evaluation file is `samples.csv`:

```bash
python evaluation.py
```

Or specify another labeled dataset:

```bash
python evaluation.py --input samples.csv --output evaluation_results.csv --matrix confusion_matrix.png
```

## Outputs

The application can export:

- complete row-level CSV results,
- complete row-level JSON results,
- analytical PDF report,
- classification report and confusion matrix when labels are available.

## Novelty positioning for a course project

The defensible novelty is **system-level integration and explainability reliability**, not a new underlying sentiment model.

A suitable presentation statement is:

> The framework integrates Transformer-based sentiment prediction with local LIME explanations, prediction uncertainty, sentiment-conflict detection, and an explanation-consistency check. It extends prediction into an auditable workflow that reports not only what sentiment was predicted, but also how confidently the model predicted it, what evidence influenced the prediction, and whether the explanation is consistent with controlled model perturbation.
