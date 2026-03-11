# =============================================================================
# evaluate.py — Etapa 8: Avaliação completa do modelo treinado
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datasets import Dataset
from transformers import Trainer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from src.config import FIG_OUTPUT_DIR, ID2LABEL, LABEL_NEGATIVE, LABEL_POSITIVE


def _ensure_fig_dir() -> None:
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)


def _get_predictions(trainer: Trainer, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Executa predições e retorna (y_pred, y_true)."""
    print("[evaluate] Executando predições no conjunto de teste...")
    predictions = trainer.predict(dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    return y_pred, y_true


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Exibe relatório completo de classificação por classe."""
    print("\n" + "=" * 60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 60)
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plota a matriz de confusão como heatmap normalizado."""
    _ensure_fig_dir()
    labels = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absoluta
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=axes[0]
    )
    axes[0].set_title("Matriz de Confusão (contagens)", fontweight="bold")
    axes[0].set_ylabel("Real")
    axes[0].set_xlabel("Predito")

    # Normalizada
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=axes[1]
    )
    axes[1].set_title("Matriz de Confusão (normalizada)", fontweight="bold")
    axes[1].set_ylabel("Real")
    axes[1].set_xlabel("Predito")

    plt.tight_layout()
    path = os.path.join(FIG_OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Salvo em: {path}")


def plot_training_curves(trainer: Trainer) -> None:
    """
    Plota curvas de Loss e F1 por época (treino e validação).
    Extrai os logs do histórico do Trainer.
    """
    _ensure_fig_dir()
    log_history = trainer.state.log_history

    # Separa logs de treino e avaliação
    train_logs = [l for l in log_history if "loss" in l and "eval_loss" not in l]
    eval_logs = [l for l in log_history if "eval_loss" in l]

    if not eval_logs:
        print("[evaluate] Nenhum log de avaliação encontrado — curvas não geradas.")
        return

    epochs_eval = [l["epoch"] for l in eval_logs]
    eval_loss = [l["eval_loss"] for l in eval_logs]
    eval_f1 = [l.get("eval_f1", None) for l in eval_logs]
    eval_acc = [l.get("eval_accuracy", None) for l in eval_logs]

    steps_train = [l["step"] for l in train_logs]
    train_loss = [l["loss"] for l in train_logs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    ax.plot(steps_train, train_loss, label="Train Loss", color="#3498db", alpha=0.6, linewidth=1)
    ax.plot(epochs_eval, eval_loss, label="Val Loss", color="#e74c3c",
            marker="o", linewidth=2, markersize=7)
    ax.set_title("Loss por Época", fontweight="bold")
    ax.set_xlabel("Época / Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # F1 / Accuracy
    ax = axes[1]
    if any(v is not None for v in eval_f1):
        ax.plot(epochs_eval, eval_f1, label="Val F1", color="#2ecc71",
                marker="o", linewidth=2, markersize=7)
    if any(v is not None for v in eval_acc):
        ax.plot(epochs_eval, eval_acc, label="Val Accuracy", color="#9b59b6",
                marker="s", linewidth=2, markersize=7, linestyle="--")
    ax.set_title("F1 e Accuracy por Época", fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("Curvas de Treinamento — DistilBERT", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Salvo em: {path}")


def show_example_predictions(
    trainer: Trainer,
    dataset: Dataset,
    df_original: pd.DataFrame,
    n_samples: int = 5,
) -> None:
    """
    Exibe exemplos de predições corretas e incorretas para
    análise qualitativa do modelo.
    """
    y_pred, y_true = _get_predictions(trainer, dataset)
    texts = df_original["Consumer complaint narrative"].iloc[-len(y_true):].reset_index(drop=True)

    correct_idx = np.where(y_pred == y_true)[0][:n_samples]
    wrong_idx = np.where(y_pred != y_true)[0][:n_samples]

    print("\n" + "=" * 60)
    print(f"EXEMPLOS CORRETOS (primeiros {n_samples})")
    print("=" * 60)
    for i in correct_idx:
        print(f"\n[Real: {ID2LABEL[y_true[i]]} | Predito: {ID2LABEL[y_pred[i]]}]")
        print(texts.iloc[i][:300] + "...")

    print("\n" + "=" * 60)
    print(f"EXEMPLOS INCORRETOS (primeiros {n_samples})")
    print("=" * 60)
    for i in wrong_idx:
        print(f"\n[Real: {ID2LABEL[y_true[i]]} | Predito: {ID2LABEL[y_pred[i]]}]")
        print(texts.iloc[i][:300] + "...")


def full_evaluation(
    trainer: Trainer,
    test_dataset: Dataset,
    df_original: pd.DataFrame = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Executa a avaliação completa: relatório, matriz de confusão,
    curvas de treinamento e exemplos de predição.

    Args:
        trainer: Trainer após treinamento.
        test_dataset: Dataset de teste tokenizado.
        df_original: DataFrame original (para exibir textos nos exemplos).

    Returns:
        Tupla (y_pred, y_true) para uso externo se necessário.
    """
    print("=" * 60)
    print("ETAPA 8 — Avaliação Completa")
    print("=" * 60)

    y_pred, y_true = _get_predictions(trainer, test_dataset)

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_training_curves(trainer)

    if df_original is not None:
        show_example_predictions(trainer, test_dataset, df_original)

    return y_pred, y_true
