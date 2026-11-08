"""Dry Bean Classifier Lab - Streamlit front end for ML Assignment 2.

Loads the five pre-trained pipelines from model/ and evaluates them on an
uploaded test CSV. No model is retrained here.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    h1 {color: #3E5C2E; letter-spacing: -0.5px;}
    div[data-testid="stMetric"] {
        background: #F1F5EC; border-left: 4px solid #6B8F4E;
        padding: 0.6rem 0.9rem; border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    pipelines = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["model_files"].items()}
    return meta, encoder, pipelines


def score_model(model, X, y, is_binary, n_classes):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    avg = "binary" if is_binary else "weighted"
    try:
        if is_binary:
            auc = roc_auc_score(y, proba[:, 1])
        else:
            auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted",
                                labels=np.arange(n_classes))
    except ValueError:
        auc = np.nan  # e.g. some class missing from the uploaded file
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


meta, encoder, pipelines = load_artifacts()
FEATURES = meta["feature_columns"]
TARGET = meta["target_column"]
CLASSES = meta["class_names"]
IS_BINARY = meta["is_binary"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🫘 Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"],
                                help=f"Must contain the {len(FEATURES)} feature columns and '{TARGET}'.")
    use_sample = st.toggle("Use bundled test_data.csv instead", value=False)
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test CSV", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    st.divider()
    model_name = st.selectbox("Choose a model", list(pipelines.keys()))
    st.caption("Models were trained offline in the notebook and are only loaded here.")

# ---------------- Header ----------------
st.title("Dry Bean Classifier Lab")
st.write(
    "Seven varieties of dry bean - Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira - "
    "identified from 16 shape measurements taken from grain images (UCI dataset #602). "
    "Upload held-out test data to see how each of the five trained models performs."
)

if uploaded is not None:
    data = pd.read_csv(uploaded)
elif use_sample and SAMPLE_CSV.exists():
    data = pd.read_csv(SAMPLE_CSV)
else:
    st.info("Upload a test CSV from the sidebar (or switch on the bundled sample) to begin.")
    with st.expander("Expected columns"):
        st.write(FEATURES + [TARGET])
    st.stop()

data.columns = [str(c).strip() for c in data.columns]
missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {missing}")
    st.stop()

data = data.dropna(subset=[TARGET])
labels = data[TARGET].astype(str).str.strip()
unknown = sorted(set(labels) - set(CLASSES))
if unknown:
    st.error(f"Unknown class labels in '{TARGET}': {unknown}. Expected: {CLASSES}")
    st.stop()

X = data[FEATURES]
y = encoder.transform(labels)

with st.expander(f"Preview uploaded data ({len(data):,} rows)"):
    st.dataframe(data.head(50), width="stretch")
    st.bar_chart(labels.value_counts())

tab_model, tab_compare, tab_pred = st.tabs(["Selected model", "Compare all models", "Predictions"])

# ---------------- Selected model ----------------
with tab_model:
    metrics, y_pred = score_model(pipelines[model_name], X, y, IS_BINARY, len(CLASSES))
    st.subheader(model_name)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=np.arange(len(CLASSES)))
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=np.arange(len(CLASSES)),
                                       target_names=CLASSES, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---------------- Comparison ----------------
with tab_compare:
    rows = {name: score_model(m, X, y, IS_BINARY, len(CLASSES))[0] for name, m in pipelines.items()}
    comp = pd.DataFrame(rows).T.round(4)
    comp.index.name = "Model"
    st.dataframe(comp.style.highlight_max(axis=0, color="#CFE3BF").format("{:.4f}"),
                 width="stretch")
    best = comp["MCC"].idxmax()
    st.success(f"Highest MCC on this data: **{best}** ({comp.loc[best, 'MCC']:.4f})")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    comp.plot(kind="bar", ax=ax2, width=0.8, colormap="summer")
    ax2.set_ylim(max(0, comp.min().min() - 0.1), 1.0)
    ax2.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig2)

# ---------------- Predictions ----------------
with tab_pred:
    out = data.copy()
    out["Predicted"] = encoder.inverse_transform(y_pred)
    out["Correct"] = out["Predicted"] == labels.values
    st.write(f"{model_name}: {out['Correct'].sum():,} of {len(out):,} rows classified correctly.")
    only_wrong = st.checkbox("Show misclassified rows only")
    st.dataframe(out[~out["Correct"]] if only_wrong else out, width="stretch")
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv")
