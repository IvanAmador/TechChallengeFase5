# =============================================================================
# label_engineering.py — Etapa 3: Criação da variável alvo (sentimento)
# =============================================================================

import pandas as pd
from src.config import LABEL_COL, LABEL_POSITIVE, LABEL_NEGATIVE


# Respostas da empresa que indicam resolução concreta para o consumidor
_POSITIVE_KEYWORDS = ("monetary relief", "non-monetary relief")


def _classify_row(row: pd.Series) -> str:
    """
    Classifica uma única linha como Positivo ou Negativo usando múltiplos sinais:

    Hierarquia de decisão:
    1. Se a resposta da empresa contém 'relief' → Positivo
       (empresa reconheceu o problema e tomou ação concreta)
    2. Se o consumidor contestou explicitamente ('Consumer disputed? == Yes') → Negativo
       (sinal forte de insatisfação, independente da resposta)
    3. Todos os demais casos (explanation, closed) → Negativo

    Nota sobre 'Consumer disputed?':
    O campo tem ~95% de NaN no dataset moderno do CFPB (coletado após 2017
    quando o campo foi descontinuado). É usado apenas como reforço quando
    disponível, nunca como condição obrigatória.
    """
    response = str(row.get("Company response to consumer", "")).lower()

    # Sinal 1 — resposta da empresa indica resolução
    if any(kw in response for kw in _POSITIVE_KEYWORDS):
        return LABEL_POSITIVE

    # Sinal 2 — consumidor contestou explicitamente
    disputed = str(row.get("Consumer disputed?", "")).strip()
    if disputed == "Yes":
        return LABEL_NEGATIVE

    # Default — apenas explicação ou fechamento sem resolução
    return LABEL_NEGATIVE


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a função de classificação ao DataFrame e adiciona a coluna
    'sentimento'. Opera in-place (retorna o mesmo DataFrame modificado).

    Args:
        df: DataFrame com pelo menos as colunas:
            - 'Company response to consumer'
            - 'Consumer disputed?' (opcional — pode ser NaN)

    Returns:
        DataFrame com a coluna 'sentimento' adicionada.
    """
    df = df.copy()
    df[LABEL_COL] = df.apply(_classify_row, axis=1)
    return df


def label_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Retorna a distribuição de sentimentos (contagem e percentual).
    Útil para verificar o balanceamento antes do treino.
    """
    counts = df[LABEL_COL].value_counts()
    pct = df[LABEL_COL].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "pct": pct.round(1)})
    print("[label_engineering] Distribuição de sentimentos:")
    print(summary.to_string())
    return counts
