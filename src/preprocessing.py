# =============================================================================
# preprocessing.py — Etapa 4: Limpeza de texto para análise exploratória
# =============================================================================
# NOTA: Este pré-processamento NÃO é aplicado ao texto que entra no DistilBERT.
# O DistilBERT usa o texto bruto via seu próprio tokenizador WordPiece.
# A limpeza aqui serve exclusivamente para:
#   - TF-IDF exploratório (Etapa 5)
#   - Word Clouds (Etapa 9)
# =============================================================================

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloads automáticos (skip se já existirem)
for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Palavras específicas do domínio CFPB que não agregam semântica
_DOMAIN_STOPWORDS = {
    "account", "company", "credit", "report", "consumer", "said",
    "told", "letter", "time", "would", "could", "also", "one",
}
_ALL_STOPWORDS = _stop_words | _DOMAIN_STOPWORDS


def clean_text(text: str, min_word_len: int = 3) -> str:
    """
    Limpa e normaliza texto de reclamação financeira para uso em
    análise exploratória (TF-IDF, word cloud).

    Etapas:
    1. Lowercase
    2. Remove tokens de anonimização CFPB ('XXXX', 'XX/XX/XXXX', etc.)
    3. Remove pontuação, números e caracteres especiais
    4. Tokeniza por espaço
    5. Remove stopwords (inglês + domínio CFPB)
    6. Lematiza cada token
    7. Filtra tokens muito curtos

    Args:
        text: Texto bruto da coluna 'Consumer complaint narrative'.
        min_word_len: Comprimento mínimo de token para manter.

    Returns:
        String limpa pronta para TF-IDF ou WordCloud.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove anonimizações (2+ X's consecutivos, com ou sem separadores)
    text = re.sub(r"\bx{2,}\b", " ", text)
    text = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", " ", text)   # datas no formato MM/DD/YYYY

    # 3. Remove caracteres não alfabéticos
    text = re.sub(r"[^a-z\s]", " ", text)

    # 4-7. Tokeniza, filtra e lematiza
    tokens = [
        _lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in _ALL_STOPWORDS and len(w) >= min_word_len
    ]

    return " ".join(tokens)


def apply_cleaning(df: pd.DataFrame, text_col: str = "Consumer complaint narrative") -> pd.DataFrame:
    """
    Aplica clean_text à coluna de texto e adiciona 'text_clean' ao DataFrame.

    Args:
        df: DataFrame com a coluna de texto.
        text_col: Nome da coluna com o texto bruto.

    Returns:
        DataFrame com coluna 'text_clean' adicionada.
    """
    print(f"[preprocessing] Limpando textos da coluna '{text_col}'...")
    df = df.copy()
    df["text_clean"] = df[text_col].apply(clean_text)

    # Estatísticas rápidas
    empty = (df["text_clean"].str.strip() == "").sum()
    avg_tokens = df["text_clean"].apply(lambda x: len(x.split())).mean()
    print(f"  Textos vazios após limpeza: {empty:,}")
    print(f"  Média de tokens por texto: {avg_tokens:.1f}")

    return df
