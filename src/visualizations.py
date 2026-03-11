# =============================================================================
# visualizations.py — Etapa 9: Word Clouds e gráficos das dores dos clientes
# =============================================================================

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from src.config import (
    LABEL_COL, LABEL_NEGATIVE,
    TOP_N_PRODUCTS, TOP_N_ISSUES,
    WORDCLOUD_MAX_WORDS, FIG_OUTPUT_DIR,
)


def _ensure_fig_dir() -> None:
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)


def plot_wordcloud_by_product(df: pd.DataFrame, text_col: str = "text_clean") -> None:
    """
    Gera Word Clouds para os TOP_N_PRODUCTS produtos mais reclamados,
    filtrando apenas sentimentos Negativos.

    Cada produto gera uma figura separada, exibida e salva como PNG.

    Args:
        df: DataFrame com colunas 'Product', 'sentimento' e text_col.
        text_col: Coluna de texto pré-processado (para word cloud limpa).
    """
    _ensure_fig_dir()
    df_neg = df[df[LABEL_COL] == LABEL_NEGATIVE].copy()
    top_products = df_neg["Product"].value_counts().head(TOP_N_PRODUCTS).index.tolist()

    print(f"[visualizations] Gerando Word Clouds para top {TOP_N_PRODUCTS} produtos...")

    for product in top_products:
        subset = df_neg[df_neg["Product"] == product][text_col].dropna()
        text = " ".join(subset)

        if not text.strip():
            print(f"  Sem dados suficientes para: {product}")
            continue

        wc = WordCloud(
            width=900,
            height=450,
            background_color="white",
            max_words=WORDCLOUD_MAX_WORDS,
            colormap="Reds",
            collocations=False,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.imshow(wc, interpolation="bilinear")
        # Nome curto do produto para o título
        short_name = product[:60] + "..." if len(product) > 60 else product
        ax.set_title(
            f"Principais Dores dos Clientes\n{short_name}",
            fontsize=13, fontweight="bold"
        )
        ax.axis("off")
        plt.tight_layout()

        safe_name = product[:40].replace("/", "-").replace(" ", "_")
        path = os.path.join(FIG_OUTPUT_DIR, f"wordcloud_{safe_name}.png")
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"  Salvo: {path}")


def plot_top_issues(df: pd.DataFrame) -> None:
    """
    Gráfico horizontal de barras com os TOP_N_ISSUES temas (Issue)
    mais frequentes nas reclamações Negativas.
    """
    _ensure_fig_dir()
    df_neg = df[df[LABEL_COL] == LABEL_NEGATIVE]
    top_issues = df_neg["Issue"].value_counts().head(TOP_N_ISSUES)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("Reds_r", n_colors=len(top_issues))
    bars = ax.barh(top_issues.index[::-1], top_issues.values[::-1], color=colors[::-1])

    # Labels nos valores
    for bar in bars:
        ax.text(
            bar.get_width() + top_issues.max() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width()):,}",
            va="center", fontsize=9,
        )

    ax.set_title(
        f"Top {TOP_N_ISSUES} Temas de Insatisfação (Reclamações Negativas)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Número de Reclamações")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.tight_layout()
    path = os.path.join(FIG_OUTPUT_DIR, "top_issues.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[visualizations] Top issues salvo em: {path}")


def plot_sentiment_by_product(df: pd.DataFrame) -> None:
    """
    Gráfico de barras empilhadas mostrando distribuição de sentimento
    (Positivo vs Negativo) para os top produtos.
    Permite identificar quais categorias têm maior taxa de insatisfação.
    """
    _ensure_fig_dir()
    top_products = df["Product"].value_counts().head(TOP_N_PRODUCTS).index.tolist()
    df_top = df[df["Product"].isin(top_products)]

    # Tabela cruzada normalizada
    pivot = (
        pd.crosstab(df_top["Product"], df_top[LABEL_COL], normalize="index") * 100
    )
    # Ordena por % negativo (mais insatisfeito primeiro)
    pivot = pivot.sort_values(LABEL_NEGATIVE, ascending=True)

    # Nomes curtos para o eixo
    short_names = {p: (p[:35] + "..." if len(p) > 35 else p) for p in pivot.index}
    pivot.index = [short_names[p] for p in pivot.index]

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color={"Negativo": "#e74c3c", "Positivo": "#2ecc71"},
        edgecolor="white",
    )
    ax.set_title(
        "Distribuição de Sentimento por Categoria de Produto (%)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Percentual (%)")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    ax.axvline(50, color="white", linestyle="--", linewidth=1, alpha=0.5)

    # Adiciona texto de % em cada barra
    for i, (_, row) in enumerate(pivot.iterrows()):
        neg_pct = row.get(LABEL_NEGATIVE, 0)
        ax.text(neg_pct / 2, i, f"{neg_pct:.1f}%", va="center", ha="center",
                fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIG_OUTPUT_DIR, "sentiment_by_product.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[visualizations] Sentimento por produto salvo em: {path}")


def plot_all(df: pd.DataFrame, text_col: str = "text_clean") -> None:
    """
    Executa todas as visualizações em sequência.
    Chame esta função no notebook na etapa final.

    Args:
        df: DataFrame com colunas 'text_clean', 'sentimento', 'Product', 'Issue'.
        text_col: Coluna de texto pré-processado.
    """
    print("=" * 60)
    print("ETAPA 9 — Visualizações das Dores dos Clientes")
    print("=" * 60)

    print("\n[visualizations] Word Clouds por produto:")
    plot_wordcloud_by_product(df, text_col=text_col)

    print("\n[visualizations] Top temas de insatisfação:")
    plot_top_issues(df)

    print("\n[visualizations] Sentimento por produto:")
    plot_sentiment_by_product(df)

    print(f"\n[visualizations] Todas as figuras salvas em '{FIG_OUTPUT_DIR}/'")
