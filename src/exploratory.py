# =============================================================================
# exploratory.py — Etapa 5: Análise exploratória (TF-IDF, distribuições)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import LABEL_COL, LABEL_POSITIVE, LABEL_NEGATIVE, FIG_OUTPUT_DIR


def _ensure_fig_dir() -> None:
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)


def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    """
    Gráfico de barras com a distribuição de sentimentos.
    Mostra contagem e percentual de Positivo vs Negativo.
    """
    _ensure_fig_dir()
    counts = df[LABEL_COL].value_counts()
    pcts = df[LABEL_COL].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=["#e74c3c", "#2ecc71"], edgecolor="white")
    for bar, (label, pct) in zip(bars, pcts.items()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1000,
            f"{bar.get_height():,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11
        )
    ax.set_title("Distribuição de Sentimentos", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentimento")
    ax.set_ylabel("Número de Reclamações")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = os.path.join(FIG_OUTPUT_DIR, "sentiment_distribution.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Salvo em: {path}")


def plot_text_length_distribution(df: pd.DataFrame) -> None:
    """
    Histograma do número de tokens por texto (usando split simples).
    Útil para validar a escolha de max_length=128 no DistilBERT.
    Marca o percentil 90 e o limite de 128 tokens.
    """
    _ensure_fig_dir()
    lengths = df["Consumer complaint narrative"].dropna().apply(lambda x: len(str(x).split()))

    p90 = np.percentile(lengths, 90)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=80, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(128, color="#e74c3c", linestyle="--", linewidth=1.5, label="max_length=128")
    ax.axvline(p90, color="#f39c12", linestyle=":", linewidth=1.5, label=f"P90 = {p90:.0f} tokens")
    ax.set_title("Distribuição do Comprimento dos Textos", fontsize=13, fontweight="bold")
    ax.set_xlabel("Número de palavras")
    ax.set_ylabel("Frequência")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_OUTPUT_DIR, "text_length_distribution.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  P90 = {p90:.0f} palavras | max_length=128 cobre {(lengths <= 128).mean()*100:.1f}% dos textos")
    print(f"  Salvo em: {path}")


def plot_tfidf_top_ngrams(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    n: int = 20,
    ngram_range: tuple = (1, 2),
) -> None:
    """
    Para cada classe (Positivo / Negativo), calcula os n tokens/bigramas
    com maior TF-IDF médio e exibe gráfico horizontal comparativo.

    Args:
        df: DataFrame com coluna 'text_clean' e 'sentimento'.
        text_col: Coluna de texto pré-processado.
        n: Top-N n-gramas por classe.
        ngram_range: Tupla (min, max) para n-gramas.
    """
    _ensure_fig_dir()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {LABEL_POSITIVE: "#2ecc71", LABEL_NEGATIVE: "#e74c3c"}

    for ax, label in zip(axes, [LABEL_NEGATIVE, LABEL_POSITIVE]):
        subset = df[df[LABEL_COL] == label][text_col].dropna()
        subset = subset[subset.str.strip() != ""]

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=5000,
            min_df=5,
        )
        tfidf_matrix = vectorizer.fit_transform(subset)
        mean_scores = tfidf_matrix.mean(axis=0).A1
        top_idx = mean_scores.argsort()[-n:][::-1]
        top_terms = [vectorizer.get_feature_names_out()[i] for i in top_idx]
        top_scores = mean_scores[top_idx]

        ax.barh(top_terms[::-1], top_scores[::-1], color=colors[label], edgecolor="white")
        ax.set_title(f"Top {n} N-gramas — {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("TF-IDF Médio")
        ax.tick_params(axis="y", labelsize=9)

    plt.suptitle("Termos mais discriminativos por sentimento", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_OUTPUT_DIR, "tfidf_top_ngrams.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Salvo em: {path}")


def run_exploratory_analysis(df: pd.DataFrame) -> None:
    """
    Executa toda a análise exploratória em sequência.
    Chame esta função no notebook após carregar e limpar o dataset.
    """
    print("=" * 60)
    print("ETAPA 5 — Análise Exploratória")
    print("=" * 60)

    print("\n[exploratory] Distribuição de sentimentos:")
    plot_sentiment_distribution(df)

    print("\n[exploratory] Comprimento dos textos:")
    plot_text_length_distribution(df)

    if "text_clean" in df.columns:
        print("\n[exploratory] Top TF-IDF n-gramas por classe:")
        plot_tfidf_top_ngrams(df)
    else:
        print("\n[exploratory] Coluna 'text_clean' não encontrada — pulando TF-IDF.")
        print("  Execute preprocessing.apply_cleaning(df) antes desta etapa.")
