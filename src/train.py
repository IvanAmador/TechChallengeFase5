# =============================================================================
# train.py — Etapa 7: Fine-tuning do DistilBERT com CustomTrainer
# =============================================================================

import os
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from src.config import (
    MODEL_NAME, NUM_LABELS, NUM_EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY, OUTPUT_DIR,
    DRIVE_CHECKPOINT_DIR, LABEL2ID, ID2LABEL, DATALOADER_NUM_WORKERS,
)


# ── Métricas para o Trainer ───────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    """
    Calcula accuracy e F1 (weighted) para cada step de avaliação.
    Usado internamente pelo Trainer.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# ── CustomTrainer com class weights ──────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Subclasse do Trainer que aplica class weights na função de loss.
    Necessário porque o dataset tem desbalanceamento (~66% Neg / 34% Pos).
    """

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── Função principal de treinamento ──────────────────────────────────────────

def train_model(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    y_train: np.ndarray = None,
) -> tuple:
    """
    Carrega o DistilBERT pré-treinado, configura o WeightedTrainer e
    executa o fine-tuning.

    Args:
        train_dataset: Dataset HuggingFace de treino (tokenizado).
        eval_dataset: Dataset HuggingFace de validação/teste (tokenizado).
        y_train: Array de labels do treino (para calcular class weights).
                 Se None, os pesos são calculados a partir do train_dataset.

    Returns:
        Tupla (trainer, model) após o treinamento.
    """
    # ── Class weights ─────────────────────────────────────────────────────────
    if y_train is None:
        y_train = np.array(train_dataset["label"])

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"[train] Class weights: { {int(c): f'{w:.3f}' for c, w in zip(classes, weights)} }")

    # ── Modelo ────────────────────────────────────────────────────────────────
    print(f"[train] Carregando modelo: '{MODEL_NAME}'")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Salvar em Drive se disponível (evita perder ao reiniciar Colab) ───────
    save_dir = DRIVE_CHECKPOINT_DIR if os.path.isdir("/content/drive") else OUTPUT_DIR
    print(f"[train] Checkpoints serão salvos em: '{save_dir}'")

    # ── Training Arguments ────────────────────────────────────────────────────
    total_steps = (len(train_dataset) // TRAIN_BATCH_SIZE) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),         # Mixed precision na T4
        optim="adamw_torch_fused",              # Otimizador fused (PyTorch ≥ 2.0): ~10% mais rápido
        dataloader_num_workers=DATALOADER_NUM_WORKERS,  # 4 CPUs em paralelo
        dataloader_pin_memory=torch.cuda.is_available(),  # Transferência CPU→GPU mais rápida
        group_by_length=True,                   # Agrupa sequências de tamanho similar → menos padding
        seed=42,
        report_to="none",                       # Desativa wandb/tensorboard por padrão
    )

    # ── Data Collator — padding dinâmico por batch ─────────────────────────────
    # Pads cada batch apenas até o comprimento máximo daquele batch (não do dataset inteiro).
    # Com sequências curtas (~69 tokens em média), evita processar centenas de tokens [PAD].
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"[train] Iniciando fine-tuning — {NUM_EPOCHS} épocas, batch={TRAIN_BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"  Total de steps: {total_steps:,} | Warmup steps: {warmup_steps:,}")
    trainer.train()

    print("[train] Treinamento concluído.")
    return trainer, model
