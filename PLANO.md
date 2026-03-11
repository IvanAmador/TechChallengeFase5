# Plano de Implementação — TechChallenge Fase 5
## Classificação de Reclamações Financeiras e Análise das Dores dos Clientes

---

## Visão Geral

Substituição do modelo LSTM treinado do zero (68.7% accuracy, 143k amostras) por um pipeline moderno baseado em **Transfer Learning com DistilBERT**, treinado sobre uma amostra balanceada de **300k registros** extraídos do dataset completo (~3.5M registros, 1.6 GB zip).

O notebook principal (`analise_reclamacoes_v2.ipynb`) orquestra etapas independentes, cada uma implementada em um módulo `.py` dedicado dentro da pasta `src/`.

---

## Estrutura de Arquivos

```
TechChallengeFase5/
├── PLANO.md                          ← este arquivo
├── analise_reclamacoes_v2.ipynb      ← notebook principal (novo)
├── analise_reclamacoes_financeiras.ipynb  ← notebook original (mantido)
├── requirements.txt                   ← atualizado com novas dependências
└── src/
    ├── config.py                     ← constantes e paths compartilhados
    ├── download_data.py              ← Etapa 1: download e extração do CSV
    ├── load_data.py                  ← Etapa 2: leitura em chunks + amostra balanceada
    ├── label_engineering.py          ← Etapa 3: criação da variável alvo
    ├── preprocessing.py              ← Etapa 4: limpeza de texto (para word clouds)
    ├── exploratory.py               ← Etapa 5: análise TF-IDF com n-gramas
    ├── hf_dataset.py                ← Etapa 6: tokenização e Dataset HuggingFace
    ├── train.py                      ← Etapa 7: fine-tuning DistilBERT
    ├── evaluate.py                   ← Etapa 8: métricas completas
    └── visualizations.py             ← Etapa 9: word clouds, gráficos, curvas
```

---

## Dataset

| Item | Detalhe |
|------|---------|
| URL | `https://files.consumerfinance.gov/ccdb/complaints.csv.zip` |
| Tamanho ZIP | ~1.6 GB |
| CSV descomprimido | ~5-6 GB |
| Registros totais | ~3.5 milhões |
| Amostra utilizada | 300.000 (150k Positivo + 150k Negativo, balanceado) |
| Split | 80% treino / 20% teste |

**Estratégia de carregamento (evita OOM no Colab):**
Leitura com `pd.read_csv(..., chunksize=100_000)`, filtrando `Consumer complaint narrative` não-nulo e concatenando apenas as colunas necessárias. Amostragem balanceada após concat.

---

## Modelo

### Escolha: `distilbert-base-uncased` (HuggingFace)

| Critério | Detalhe |
|---------|---------|
| Parâmetros | 66M (40% menor que BERT) |
| Velocidade | 60% mais rápido que BERT, mantém ~97% da performance |
| GPU RAM necessária | ~4-5 GB de 15 GB disponíveis no T4 |
| Precision | FP16 (mixed precision) para ~50% mais velocidade no T4 |
| Max sequence length | 128 tokens (economiza memória; textos mais longos são truncados) |
| Epochs | 3 |
| Batch size | 32 (treino) / 64 (avaliação) |
| Optimizer | AdamW (padrão HuggingFace Trainer) |
| Scheduler | Linear warmup (10% dos steps) |

**Por que não FinBERT:**
FinBERT foi treinado em notícias financeiras e relatórios SEC (domínio formal). As narrativas do CFPB são textos escritos por consumidores (domínio conversacional). Fine-tuning de DistilBERT diretamente no dataset de reclamações supera FinBERT zero-shot nesta tarefa.

**Por que não LSTM do zero:**
Com 143k amostras, o LSTM atingiu apenas 68.7% — embeddings aleatórios sem conhecimento linguístico prévio. Escalar para 300k no LSTM exigiria ~50 min/época, tornando a sessão gratuita do Colab inviável.

---

## Etapas Detalhadas

### Etapa 1 — Download e Extração (`src/download_data.py`)

- `!wget` da URL oficial do CFPB diretamente no ambiente Colab
- Extração do ZIP com `zipfile` ou `!unzip`
- Verificação do arquivo extraído (tamanho, primeiras linhas)
- Tempo estimado: ~15 min no Colab gratuito

```python
# Exemplo de uso no notebook:
from src.download_data import download_and_extract
csv_path = download_and_extract()
```

---

### Etapa 2 — Carregamento e Amostragem (`src/load_data.py`)

- Leitura em chunks de 100k linhas com `pd.read_csv(chunksize=100_000)`
- Colunas selecionadas: `Consumer complaint narrative`, `Company response to consumer`, `Consumer disputed?`, `Product`, `Issue`, `Timely response?`
- Filtro: narrativa não-nula e não-vazia
- Criação de labels provisórias por chunk (para balancear antes do concat)
- Amostra final: 150k Positivo + 150k Negativo = 300k total
- Tempo estimado: ~8 min

```python
from src.load_data import load_balanced_sample
df = load_balanced_sample(csv_path, n_per_class=150_000)
```

---

### Etapa 3 — Label Engineering (`src/label_engineering.py`)

**Lógica multi-sinal (mais robusta que o original):**

| Condição | Label |
|---------|-------|
| Response contém `'monetary relief'` | Positivo |
| Response contém `'non-monetary relief'` | Positivo |
| `Consumer disputed? == 'Yes'` (quando disponível) | Negativo |
| Response == `'Closed with explanation'` | Negativo |
| Demais casos | Negativo |

**Notas importantes:**
- `Consumer disputed?` tem ~95% de NaN no dataset — usado apenas como reforço quando disponível, nunca como condição obrigatória
- Excluídos registros com response `'In progress'` ou `'Untimely response'` (ambíguo)
- Distribuição final verificada para garantir balanço

```python
from src.label_engineering import apply_labels
df = apply_labels(df)
```

---

### Etapa 4 — Pré-processamento de Texto (`src/preprocessing.py`)

**Usado apenas para análise exploratória e visualizações (word clouds).**
O DistilBERT recebe o texto bruto — ele tem seu próprio tokenizador WordPiece que não precisa de limpeza manual.

- Remoção de tokens `XXXX` (anonimização CFPB)
- Lowercase, remoção de pontuação e números
- Remoção de stopwords (NLTK)
- Lematização (WordNetLemmatizer)

```python
from src.preprocessing import clean_text
df['text_clean'] = df['Consumer complaint narrative'].apply(clean_text)
```

---

### Etapa 5 — Análise Exploratória (`src/exploratory.py`)

- Distribuição de sentimentos por produto financeiro
- **TF-IDF com n-gramas (1,2)** — identifica bigramas mais discriminativos por classe
- Top-20 unigramas e bigramas para Negativo vs Positivo
- Distribuição do comprimento dos textos (para validar `max_length=128`)

```python
from src.exploratory import run_exploratory_analysis
run_exploratory_analysis(df)
```

---

### Etapa 6 — Tokenização e Dataset HuggingFace (`src/hf_dataset.py`)

- Tokenização com `DistilBertTokenizerFast`
- `max_length=128`, `truncation=True`, `padding='max_length'`
- Conversão para `datasets.Dataset` (eficiente em memória)
- Split treino/teste 80/20 estratificado

```python
from src.hf_dataset import prepare_hf_datasets
train_dataset, test_dataset = prepare_hf_datasets(df)
```

---

### Etapa 7 — Treinamento (`src/train.py`)

- `DistilBertForSequenceClassification` com `num_labels=2`
- `TrainingArguments` com:
  - `fp16=True` (mixed precision)
  - `evaluation_strategy='epoch'`
  - `load_best_model_at_end=True`
  - `metric_for_best_model='f1'`
- Class weights aplicados via `CustomTrainer` (subclasse de `Trainer`)
- Salvamento do modelo em `/content/drive/MyDrive/model_checkpoint/` (Google Drive)

```python
from src.train import train_model
trainer, model = train_model(train_dataset, test_dataset)
```

---

### Etapa 8 — Avaliação (`src/evaluate.py`)

Métricas completas no conjunto de teste:

- **Accuracy**
- **F1-score** (weighted e por classe)
- **Precision e Recall** por classe
- **Confusion Matrix** (heatmap)
- **Curvas de Loss e F1** por época (treino vs validação)
- Exemplos de predição (5 corretos + 5 incorretos)

```python
from src.evaluate import full_evaluation
full_evaluation(trainer, test_dataset, model)
```

---

### Etapa 9 — Visualizações das Dores dos Clientes (`src/visualizations.py`)

- **Word Cloud** por categoria de produto (top 5 produtos) — filtrando sentimento Negativo
- **Gráfico de frequência** dos 10 principais temas (`Issue`) nas reclamações negativas
- **Gráfico de distribuição** de sentimento por produto financeiro
- Todos os gráficos salvos como PNG para uso no vídeo de apresentação

```python
from src.visualizations import plot_all
plot_all(df)
```

---

## Estimativa de Tempo (Colab Gratuito — T4)

| Etapa | Tempo estimado |
|-------|---------------|
| Download + extração (1.6 GB zip) | ~15 min |
| Leitura em chunks + amostragem | ~8 min |
| Label engineering + pré-processamento | ~5 min |
| Análise exploratória (TF-IDF) | ~3 min |
| Tokenização 300k amostras | ~5 min |
| Fine-tuning DistilBERT (3 épocas) | ~45 min |
| Avaliação + visualizações | ~5 min |
| **Total** | **~86 min** |

Cabe confortavelmente numa sessão gratuita (~3-4h).

---

## Uso de Memória

| Componente | GPU RAM (T4 — 15 GB) |
|-----------|----------------------|
| DistilBERT model (FP16) | ~260 MB |
| Batch 32 × 128 tokens | ~1.5 GB |
| Optimizer states (AdamW) | ~500 MB |
| Gradientes | ~300 MB |
| **Total estimado** | **~3-4 GB de 15 GB** |

---

## `requirements.txt` Atualizado

```
pandas
numpy
matplotlib
seaborn
nltk
wordcloud
scikit-learn
transformers
torch
datasets
accelerate
```

---

## Entregáveis do Desafio — Checklist

| Requisito | Implementado em |
|-----------|----------------|
| ✅ Limpeza de texto (pontuação, stopwords, lematização) | `src/preprocessing.py` |
| ✅ Vetorização (TF-IDF n-gramas) | `src/exploratory.py` |
| ✅ Variável alvo Positivo/Negativo | `src/label_engineering.py` |
| ✅ Modelo Deep Learning (DistilBERT) | `src/train.py` |
| ✅ Métricas de validação (F1, Precision, Recall, CM) | `src/evaluate.py` |
| ✅ Word cloud por categoria de produto | `src/visualizations.py` |
| ✅ Gráfico de frequência de temas negativos | `src/visualizations.py` |
| ✅ Repositório GitHub com pipeline completo | estrutura `src/` |

---

## Decisões de Design

| Decisão | Motivo |
|---------|--------|
| DistilBERT em vez de LSTM | +20pp de accuracy, encaixa no Colab gratuito |
| Sem FinBERT | Domínio diferente (notícias vs narrativas de consumidores) |
| Amostra 300k em vez do dataset completo | RAM do Colab (12.7 GB CPU) não comporta 5-6 GB de CSV |
| max_length=128 em vez de 512 | Economiza 4x de memória GPU; histograma de tokens valida a escolha |
| FP16 | +50% velocidade no T4 sem perda de qualidade |
| CustomTrainer com class_weight | Dataset desbalanceado (66% Neg / 33% Pos) |
| Módulos `.py` separados | Separação de responsabilidades, fácil manutenção e reaproveitamento |
| Salvar modelo no Google Drive | Evita perder checkpoint ao reiniciar sessão Colab |
