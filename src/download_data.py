# =============================================================================
# download_data.py — Etapa 1: Download e extração do dataset completo do CFPB
# =============================================================================

import os
import zipfile
import urllib.request
from src.config import DATASET_URL, ZIP_FILENAME, CSV_FILENAME


def download_and_extract(dest_dir: str = ".") -> str:
    """
    Faz o download do dataset completo de reclamações financeiras da CFPB
    e extrai o CSV, retornando o caminho do arquivo CSV extraído.

    Em ambiente Colab recomenda-se usar !wget para ter barra de progresso
    visual. Esta função serve como fallback programático.

    Args:
        dest_dir: Diretório de destino para download e extração.

    Returns:
        Caminho completo do arquivo CSV extraído.
    """
    zip_path = os.path.join(dest_dir, ZIP_FILENAME)
    csv_path = os.path.join(dest_dir, CSV_FILENAME)

    # ── Download ──────────────────────────────────────────────────────────────
    if os.path.exists(csv_path):
        print(f"[download_data] CSV já existe em '{csv_path}'. Pulando download.")
        return csv_path

    if not os.path.exists(zip_path):
        print(f"[download_data] Baixando dataset de:\n  {DATASET_URL}")
        print("  Isso pode levar ~15 minutos no Colab gratuito...")
        urllib.request.urlretrieve(DATASET_URL, zip_path, _progress_hook)
        print(f"\n[download_data] Download concluído: {zip_path}")
    else:
        print(f"[download_data] ZIP já existe em '{zip_path}'. Pulando download.")

    # ── Extração ──────────────────────────────────────────────────────────────
    print(f"[download_data] Extraindo ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"  Arquivos no ZIP: {names}")
        zf.extractall(dest_dir)

    # O arquivo extraído pode ter nome diferente — localiza o CSV
    for name in os.listdir(dest_dir):
        if name.endswith(".csv"):
            extracted = os.path.join(dest_dir, name)
            if extracted != csv_path:
                os.rename(extracted, csv_path)
            break

    size_gb = os.path.getsize(csv_path) / 1e9
    print(f"[download_data] CSV extraído: '{csv_path}' ({size_gb:.2f} GB)")
    return csv_path


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Exibe progresso do download via urllib."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct // 2)
        print(f"\r  [{'=' * bar}{' ' * (50 - bar)}] {pct:.1f}%", end="", flush=True)
