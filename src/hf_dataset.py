# =============================================================================
# hf_dataset.py — Etapa 6: Tokenização e preparação do Dataset HuggingFace
# =============================================================================

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from src.config import (
    MODEL_NAME, MAX_LENGTH, LABEL_COL, LABEL2ID, SEED
)


def _load_tokenizer() -> DistilBertTokenizerFast:
    print(f"[hf_dataset] Carregando tokenizador: '{MODEL_NAME}'")
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


def _tokenize_batch(batch: dict, tokenizer: DistilBertTokenizerFast) -> dict:
    """Função de tokenização aplicada em batches via .map().
    Sem padding aqui — o DataCollatorWithPadding padeia dinamicamente por batch,
    reduzindo tokens desnecessários e acelerando o treinamento.
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def prepare_hf_datasets(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[Dataset, Dataset]:
    """
    Prepara os datasets de treino e teste no formato HuggingFace,
    prontos para uso com o Trainer.

    Etapas:
    1. Seleciona e renomeia colunas necessárias
    2. Converte labels string → int (LABEL2ID)
    3. Split estratificado treino/teste
    4. Converte para datasets.Dataset
    5. Tokeniza em batch (mais eficiente que linha a linha)
    6. Remove colunas desnecessárias e define formato PyTorch

    Args:
        df: DataFrame com colunas 'Consumer complaint narrative' e 'sentimento'.
        test_size: Proporção do conjunto de teste (padrão: 0.2).

    Returns:
        Tupla (train_dataset, test_dataset) no formato HuggingFace Dataset.
    """
    tokenizer = _load_tokenizer()

    # ── Prepara colunas ───────────────────────────────────────────────────────
    data = pd.DataFrame({
        "text": df["Consumer complaint narrative"].astype(str),
        "label": df[LABEL_COL].map(LABEL2ID).astype(int),
    }).dropna()

    print(f"[hf_dataset] Registros válidos para treino: {len(data):,}")
    print(f"  Distribuição de labels: {data['label'].value_counts().to_dict()}")

    # ── Split estratificado ───────────────────────────────────────────────────
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=SEED,
        stratify=data["label"],
    )

    print(f"  Treino: {len(train_df):,} | Teste: {len(test_df):,}")

    # ── Converte para HuggingFace Dataset ─────────────────────────────────────
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    # ── Tokenização em batch ──────────────────────────────────────────────────
    print("[hf_dataset] Tokenizando conjunto de treino...")
    train_ds = train_ds.map(
        lambda batch: _tokenize_batch(batch, tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=2,
        desc="Tokenizando treino",
    )

    print("[hf_dataset] Tokenizando conjunto de teste...")
    test_ds = test_ds.map(
        lambda batch: _tokenize_batch(batch, tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=2,
        desc="Tokenizando teste",
    )

    # ── Formato PyTorch ───────────────────────────────────────────────────────
    cols_to_keep = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.select_columns(cols_to_keep)
    test_ds = test_ds.select_columns(cols_to_keep)

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    print("[hf_dataset] Datasets prontos para o Trainer.")
    return train_ds, test_ds
