"""Professional Streamlit dashboard for the exported DDoS classifier."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_tl.app_helpers import (  # noqa: E402
    build_prediction_frame,
    evaluate_labeled_frame,
    load_uploaded_csv,
    transform_for_inference,
    validate_input_columns,
)
from ddos_tl.config import load_config, resolve  # noqa: E402
from ddos_tl.data.preprocessing import Preprocessor  # noqa: E402
from ddos_tl.evaluation.plots import plot_confusion_matrix  # noqa: E402
from ddos_tl.inference import get_device, predict_proba  # noqa: E402
from ddos_tl.models.mlp import TabularMLP  # noqa: E402
from ddos_tl.utils.io import load_json  # noqa: E402

st.set_page_config(page_title="DDoS Attack Detection", page_icon="🛡️", layout="wide")
CFG = load_config("configs/default.yaml")
MODELS = resolve(CFG["paths"]["models"])
MAX_ROWS = CFG["app"]["max_rows"]


@st.cache_resource
def load_artifacts():
    meta = load_json(MODELS / "best_meta.json")
    if len(meta["feature_columns"]) != 76 or len(meta["class_names"]) != 18:
        raise ValueError("Le modèle doit utiliser exactement 76 features et 18 classes.")
    arch = meta["arch"]
    if arch["in_dim"] != len(meta["feature_columns"]):
        raise ValueError("La dimension d'entrée ne correspond pas aux features.")
    model = TabularMLP(arch["in_dim"], len(meta["class_names"]), arch["hidden"], arch["dropout"])
    model.load_state_dict(torch.load(MODELS / "best_model.pt", map_location="cpu", weights_only=True))
    prep = Preprocessor.from_dict(meta["preprocessor"])
    if prep.feature_columns != meta["feature_columns"]:
        raise ValueError("L'ordre du preprocessing ne correspond pas aux métadonnées.")
    return model, prep, meta


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def render_chart(title: str, data: pd.DataFrame, kind: str = "bar") -> None:
    st.markdown(f"**{title}**")
    try:
        import plotly.express as px
        if kind == "bar":
            figure = px.bar(data, x=data.index, y=data.columns[0], color=data.columns[0])
        else:
            figure = px.histogram(data, x=data.columns[0], nbins=20)
        figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figure, use_container_width=True)
    except ImportError:
        if kind == "bar":
            st.bar_chart(data)
        else:
            st.line_chart(data)


def main() -> None:
    st.markdown("""
    <style>
    .main-title { color:#0B1F3A; font-size:2.2rem; font-weight:700; margin-bottom:0; }
    .subtitle { color:#52708F; font-size:1.05rem; margin-top:0; }
    .status { background:#E8F5EE; color:#16734A; border-radius:999px;
              padding:0.35rem 0.75rem; font-weight:600; display:inline-block; }
    [data-testid="stMetric"] { background:#F5F8FC; border:1px solid #DCE7F3;
                                border-radius:12px; padding:0.8rem; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="main-title">DDoS Attack Detection & Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep Learning and Transfer Learning · CIC-DDoS2019</p>',
                unsafe_allow_html=True)
    st.markdown('<span class="status">● Modèle chargé</span>', unsafe_allow_html=True)

    if not (MODELS / "best_model.pt").exists() or not (MODELS / "best_meta.json").exists():
        st.error("Le modèle de production est absent. Exportez d'abord best_model.pt et best_meta.json.")
        st.stop()
    try:
        model, prep, meta = load_artifacts()
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        st.error(f"Chargement du modèle impossible : {exc}")
        st.stop()

    class_names = meta["class_names"]
    with st.sidebar:
        st.header("Configuration du modèle")
        st.success("Modèle opérationnel")
        st.write(f"**Stratégie :** {meta.get('strategy', 'Scratch')}")
        st.write(f"**Architecture :** {meta['arch']['in_dim']} → {' → '.join(map(str, meta['arch']['hidden']))} → {len(class_names)}")
        st.write(f"**Features :** {len(meta['feature_columns'])}")
        st.write(f"**Classes :** {len(class_names)}")
        st.write(f"**Macro F1 test :** {meta.get('test_metrics', {}).get('macro_f1', 0):.4f}")
        st.caption("Le score affiché est le score historique du modèle exporté, pas celui du fichier importé.")
        if st.button("Réinitialiser la session"):
            st.cache_resource.clear()
            st.rerun()

    uploaded = st.file_uploader("Importer un CSV de flux réseau CICFlowMeter", type=["csv"])
    if uploaded is None:
        st.info("Importez un fichier CSV pour lancer la classification.")
        return
    try:
        frame = load_uploaded_csv(uploaded, MAX_ROWS, meta.get("feature_alignment"))
        missing = validate_input_columns(frame, prep)
        if missing:
            st.error(f"Colonnes manquantes ({len(missing)}) : {', '.join(missing[:10])}")
            return
        X, quality = transform_for_inference(frame, prep)
    except ValueError as exc:
        st.error(str(exc))
        return

    if quality["invalid_cells"]:
        st.warning(f"{quality['invalid_cells']} valeur(s) NaN, infinie(s) ou non numérique(s) ont été imputées par les médianes du train.")
    st.caption(f"{len(frame):,} flows importés · maximum {MAX_ROWS:,} · {frame.shape[1]} colonnes")
    with st.expander("Aperçu des données", expanded=False):
        st.dataframe(frame.head(20), use_container_width=True)

    try:
        probabilities = predict_proba(model, X, get_device("auto"))
        predictions = build_prediction_frame(frame, probabilities, class_names)
    except (RuntimeError, ValueError) as exc:
        st.error(f"Inférence impossible : {exc}")
        return

    predicted = predictions["Classe prédite"]
    benign_count = int((predicted == "Benign").sum())
    kpi = st.columns(5)
    kpi[0].metric("Total Network Flows", f"{len(frame):,}")
    kpi[1].metric("Benign Flows", f"{benign_count:,}")
    kpi[2].metric("DDoS Flows", f"{len(frame) - benign_count:,}")
    kpi[3].metric("Classes prédites", str(predicted.nunique()))
    kpi[4].metric("Average Confidence", f"{predictions['Confidence'].mean():.2%}")

    st.subheader("Vue d'ensemble des prédictions")
    counts = predicted.value_counts().reindex(class_names, fill_value=0)
    left, right = st.columns(2)
    with left:
        render_chart("Répartition Benign / DDoS", pd.DataFrame({
            "Flows": [benign_count, len(frame) - benign_count]
        }, index=["Benign", "DDoS"]))
    with right:
        render_chart("Distribution des 18 classes", counts.to_frame("Flows"))
    left, right = st.columns(2)
    with left:
        render_chart("Top 5 des attaques détectées", counts.drop("Benign").sort_values(ascending=False).head(5).to_frame("Flows"))
    with right:
        render_chart("Distribution des niveaux de confiance", predictions[["Confidence"]], "hist")

    st.subheader("Table des prédictions")
    filters = st.columns([2, 1, 1])
    search = filters[0].text_input("Rechercher un flow ou une classe")
    class_filter = filters[1].selectbox("Filtrer par classe", ["Toutes"] + class_names)
    sort_desc = filters[2].checkbox("Confidence décroissante", value=True)
    table = predictions
    if class_filter != "Toutes":
        table = table[table["Classe prédite"] == class_filter]
    if search:
        mask = table.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False))
        table = table[mask.any(axis=1)]
    table = table.sort_values("Confidence", ascending=not sort_desc)
    st.dataframe(table[["Flow index", "Classe prédite", "Confidence", "Niveau"]].head(1000),
                 use_container_width=True)
    st.download_button("Télécharger les prédictions CSV",
                       predictions[["Flow index", "Classe prédite", "Confidence", "Niveau"]].to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")
    st.download_button("Télécharger les probabilités CSV",
                       predictions.to_csv(index=False).encode(), "probabilities.csv", "text/csv")

    label_col = CFG["data"]["label_column"]
    indices = predictions["Classe prédite"].map({name: i for i, name in enumerate(class_names)}).to_numpy()
    metrics, unknown = evaluate_labeled_frame(frame, indices, class_names, label_col)
    if label_col not in frame.columns:
        st.info("Aucune colonne Label : les métriques d'évaluation ne sont pas calculées.")
        return
    if unknown:
        st.warning(f"{unknown} label(s) inconnu(s) ignoré(s) pour l'évaluation.")
    if metrics is None:
        st.warning("Aucun label reconnu : métriques non calculées.")
        return
    st.subheader("Métriques sur le fichier importé")
    cols = st.columns(5)
    for col, label, key in zip(cols, ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"],
                               ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]):
        col.metric(label, f"{metrics[key]:.4f}")
    st.dataframe(pd.DataFrame(metrics["per_class"]).T, use_container_width=True)
    st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"], class_names))
    st.download_button("Télécharger le rapport JSON",
                       pd.Series(metrics).to_json().encode(), "classification_report.json", "application/json")


if __name__ == "__main__":
    main()
