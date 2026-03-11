# =============================================================================
# load_data.py — Etapa 2: Leitura em chunks e amostra balanceada
# =============================================================================

import pandas as pd
from src.config import USECOLS, CHUNK_SIZE, N_PER_CLASS, LABEL_POSITIVE, LABEL_NEGATIVE
from src.label_engineering import apply_labels


def load_balanced_sample(csv_path: str, n_per_class: int = N_PER_CLASS) -> pd.DataFrame:
    """
    Lê o CSV completo em chunks de tamanho controlado, aplica as labels
    em cada chunk e constrói uma amostra final balanceada de
    (n_per_class × 2) registros.

    Estratégia de memória:
    - Carrega apenas as colunas necessárias (USECOLS)
    - Processa e descarta cada chunk após rotulagem
    - Acumula apenas os registros com narrativa válida
    - Amostra balanceada no final

    Args:
        csv_path: Caminho do arquivo complaints.csv extraído.
        n_per_class: Número de amostras por classe (padrão: 150_000).

    Returns:
        DataFrame balanceado com colunas originais + 'sentimento'.
    """
    print(f"[load_data] Lendo '{csv_path}' em chunks de {CHUNK_SIZE:,} linhas...")
    print(f"  Colunas carregadas: {USECOLS}")

    positivos = []
    negativos = []
    total_lidos = 0
    chunk_idx = 0

    # Respostas ambíguas que devemos excluir
    RESPOSTAS_EXCLUIDAS = {"In progress", "Untimely response"}

    reader = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE,
        engine="python",
        on_bad_lines="skip",
    )

    for chunk in reader:
        chunk_idx += 1
        total_lidos += len(chunk)

        # Filtros básicos
        chunk = chunk.dropna(subset=["Consumer complaint narrative"])
        chunk = chunk[chunk["Consumer complaint narrative"].str.strip() != ""]
        chunk = chunk[
            ~chunk["Company response to consumer"].isin(RESPOSTAS_EXCLUIDAS)
        ]

        # Aplica labels
        chunk = apply_labels(chunk)

        # Acumula por classe (para depois balancear)
        pos = chunk[chunk["sentimento"] == LABEL_POSITIVE]
        neg = chunk[chunk["sentimento"] == LABEL_NEGATIVE]
        positivos.append(pos)
        negativos.append(neg)

        pos_total = sum(len(p) for p in positivos)
        neg_total = sum(len(n) for n in negativos)
        print(
            f"  Chunk {chunk_idx:3d} | Lidos: {total_lidos:>9,} | "
            f"Pos: {pos_total:>7,} | Neg: {neg_total:>7,}",
            end="\r",
        )

        # Para cedo se já temos amostras suficientes
        if pos_total >= n_per_class and neg_total >= n_per_class:
            print(f"\n  Amostra suficiente atingida no chunk {chunk_idx}. Parando.")
            break

    print(f"\n[load_data] Total de linhas lidas do CSV: {total_lidos:,}")

    # ── Balanceamento ─────────────────────────────────────────────────────────
    df_pos = pd.concat(positivos, ignore_index=True)
    df_neg = pd.concat(negativos, ignore_index=True)

    print(f"  Positivos disponíveis: {len(df_pos):,}")
    print(f"  Negativos disponíveis: {len(df_neg):,}")

    # Garante que não solicitamos mais do que o disponível
    n_pos = min(n_per_class, len(df_pos))
    n_neg = min(n_per_class, len(df_neg))

    df_pos = df_pos.sample(n=n_pos, random_state=42)
    df_neg = df_neg.sample(n=n_neg, random_state=42)

    df = pd.concat([df_pos, df_neg], ignore_index=True).sample(frac=1, random_state=42)

    print(f"[load_data] Dataset final: {len(df):,} registros ({n_pos:,} Pos + {n_neg:,} Neg)")
    return df
