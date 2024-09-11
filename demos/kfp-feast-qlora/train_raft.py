from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("json", data_files="raft_training_data.jsonl", split="train")

TRAINING_OUTFILE = "raft_training_data.jsonl"
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PRETRAINED_MODEL = "BEE-spoke-data/smol_llama-101M-GQA"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL)


response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

if torch.backends.mps.is_built():
    device = torch.device("mps")

    from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer")

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=120)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

