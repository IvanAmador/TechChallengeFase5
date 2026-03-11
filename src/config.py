# =============================================================================
# config.py — Constantes e configurações compartilhadas entre todos os módulos
# =============================================================================

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
ZIP_FILENAME = "complaints.csv.zip"
CSV_FILENAME = "complaints.csv"

# Colunas necessárias (evita carregar o CSV inteiro na memória)
USECOLS = [
    "Consumer complaint narrative",
    "Company response to consumer",
    "Consumer disputed?",
    "Timely response?",
    "Product",
    "Issue",
]

# Tamanho de cada chunk na leitura do CSV
CHUNK_SIZE = 100_000

# Amostras por classe no dataset balanceado final
N_PER_CLASS = 150_000  # 150k Positivo + 150k Negativo = 300k total

# ── Modelo ────────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128       # Tokens por sequência (truncation/padding)
NUM_LABELS = 2

# ── Treinamento ───────────────────────────────────────────────────────────────
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 64       # T4 16GB suporta batch=64 com DistilBERT fp16
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1          # 10% dos steps para warmup linear
WEIGHT_DECAY = 0.01
SEED = 42
DATALOADER_NUM_WORKERS = 4  # Colab tem 4 CPUs; paraleliza carregamento de dados

# ── Caminhos de saída ─────────────────────────────────────────────────────────
OUTPUT_DIR = "./results"
# Caminho no Google Drive para salvar checkpoints (evita perder ao reiniciar)
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/model_checkpoint"

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_COL = "sentimento"
LABEL_POSITIVE = "Positivo"
LABEL_NEGATIVE = "Negativo"
LABEL2ID = {LABEL_NEGATIVE: 0, LABEL_POSITIVE: 1}
ID2LABEL = {0: LABEL_NEGATIVE, 1: LABEL_POSITIVE}

# ── Visualizações ─────────────────────────────────────────────────────────────
TOP_N_PRODUCTS = 5      # Produtos para word cloud
TOP_N_ISSUES = 10       # Temas para gráfico de frequência
WORDCLOUD_MAX_WORDS = 100
FIG_OUTPUT_DIR = "./figures"
