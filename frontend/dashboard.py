"""Training and evaluation dashboard (reads actual artifacts from results/)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None

from frontend.components import load_json_artifact, read_text_artifact


def _show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Artifact unavailable: `{path.name}` not found in `results/`.")


def render_dashboard(results_dir: Path) -> None:
    history = load_json_artifact(str(results_dir / "training_history.json"))
    metrics = load_json_artifact(str(results_dir / "evaluation_metrics.json"))
    eval_txt = read_text_artifact(str(results_dir / "evaluation_report.txt"))
    cls_txt = read_text_artifact(str(results_dir / "classification_report.txt"))

    st.markdown("#### Evaluation metrics (from saved artifacts)")
    if metrics:
        acc = float(metrics.get("test_accuracy", 0.0))
        loss = float(metrics.get("test_loss", 0.0))
        prec = float(metrics.get("macro_precision", 0.0))
        rec = float(metrics.get("macro_recall", 0.0))
        f1 = float(metrics.get("macro_f1", 0.0))
        samples = metrics.get("test_samples", 10000)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test accuracy", f"{acc * 100:.2f}%")
        c2.metric("Test loss", f"{loss:.6f}")
        c3.metric("Macro precision", f"{prec * 100:.2f}%")
        c4.metric("Macro recall", f"{rec * 100:.2f}%")
        c5.metric("Macro F1-score", f"{f1 * 100:.2f}%")
        st.caption(f"Source: `results/evaluation_metrics.json` · Test samples: {samples}")

        per_class = metrics.get("per_class_accuracy")
        if isinstance(per_class, dict) and per_class:
            digits = list(range(10))
            vals = [float(per_class.get(str(d), 0.0)) * 100.0 for d in digits]
            if go is not None:
                fig = go.Figure(go.Bar(x=[str(d) for d in digits], y=vals, marker_color="#818cf8", text=[f"{v:.2f}%" for v in vals], textposition="outside"))
                fig.update_layout(title="Per-class accuracy (test set)", xaxis_title="Digit", yaxis_title="Accuracy (%)", yaxis=dict(range=[0, 105]), height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                st.markdown("**Per-class accuracy (test set)**")
                st.bar_chart(pd.DataFrame({"Accuracy (%)": vals}, index=[str(d) for d in digits]))
    else:
        st.warning("`results/evaluation_metrics.json` not found. Run `python evaluate.py` to generate evaluation artifacts.")

    st.markdown("#### Training metrics and curves")
    if history:
        epochs_completed = history.get("epochs_completed", len(history.get("accuracy", [])))
        train_acc = history.get("accuracy", [])
        val_acc = history.get("val_accuracy", [])
        train_loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])
        final_train = train_acc[-1] if train_acc else None
        final_val = val_acc[-1] if val_acc else None
        c1, c2, c3 = st.columns(3)
        c1.metric("Completed epochs", str(epochs_completed))
        c2.metric("Final training accuracy", f"{final_train * 100:.2f}%" if final_train is not None else "n/a")
        c3.metric("Final validation accuracy", f"{final_val * 100:.2f}%" if final_val is not None else "n/a")
        st.caption("Source: `results/training_history.json`")

        if train_acc and val_acc:
            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=[v * 100 for v in train_acc], mode="lines+markers", name="Training accuracy", line=dict(color="#38bdf8")))
                fig.add_trace(go.Scatter(y=[v * 100 for v in val_acc], mode="lines+markers", name="Validation accuracy", line=dict(color="#a78bfa")))
                fig.update_layout(title="Training vs validation accuracy", xaxis_title="Epoch (0-indexed)", yaxis_title="Accuracy (%)", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                st.markdown("**Training vs validation accuracy (%)**")
                st.line_chart(pd.DataFrame({"Training": [v * 100 for v in train_acc], "Validation": [v * 100 for v in val_acc]}))

        if train_loss and val_loss:
            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=train_loss, mode="lines+markers", name="Training loss", line=dict(color="#38bdf8")))
                fig.add_trace(go.Scatter(y=val_loss, mode="lines+markers", name="Validation loss", line=dict(color="#f97316")))
                fig.update_layout(title="Training vs validation loss", xaxis_title="Epoch (0-indexed)", yaxis_title="Loss", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                st.markdown("**Training vs validation loss**")
                st.line_chart(pd.DataFrame({"Training": train_loss, "Validation": val_loss}))

        # Tabular view
        rows = []
        for i in range(epochs_completed):
            rows.append({
                "Epoch": i + 1,
                "Train acc": train_acc[i] if i < len(train_acc) else None,
                "Val acc": val_acc[i] if i < len(val_acc) else None,
                "Train loss": train_loss[i] if i < len(train_loss) else None,
                "Val loss": val_loss[i] if i < len(val_loss) else None,
                "LR": history.get("learning_rate", [None] * epochs_completed)[i] if i < len(history.get("learning_rate", [])) else None,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.warning("`results/training_history.json` not found. Run `python train.py` to generate training artifacts.")

    st.markdown("#### Saved result figures")
    c1, c2 = st.columns(2)
    with c1:
        _show_image_if_exists(results_dir / "training_accuracy.png", "Training vs validation accuracy")
        _show_image_if_exists(results_dir / "confusion_matrix.png", "Confusion matrix (test set)")
    with c2:
        _show_image_if_exists(results_dir / "training_loss.png", "Training vs validation loss")
        _show_image_if_exists(results_dir / "sample_predictions.png", "Sample predictions")

    _show_image_if_exists(results_dir / "mnist_samples.png", "MNIST sample images")

    st.markdown("#### Text reports")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Evaluation report**")
        if eval_txt:
            st.code(eval_txt, language="text")
        else:
            st.info("`results/evaluation_report.txt` not found.")
    with t2:
        st.markdown("**Classification report**")
        if cls_txt:
            st.code(cls_txt, language="text")
        else:
            st.info("`results/classification_report.txt` not found.")
