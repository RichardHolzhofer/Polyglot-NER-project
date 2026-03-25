import os
import argparse
import numpy as np
import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import wandb

# Load metric
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds, label_names):
    """
    Detailed, portfolio-ready metrics breakdown including per-entity scores.
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    
    # 1. Start with overall results
    results = {
        "overall_precision": all_metrics["overall_precision"],
        "overall_recall": all_metrics["overall_recall"],
        "overall_f1": all_metrics["overall_f1"],
        "overall_accuracy": all_metrics["overall_accuracy"],
    }
    
    # 2. Extract metrics for EACH entity type (PER, ORG, LOC, etc.)
    for key, value in all_metrics.items():
        if key not in ["overall_precision", "overall_recall", "overall_f1", "overall_accuracy"]:
            results[f"{key}_f1"] = value["f1"]
            results[f"{key}_precision"] = value["precision"]
            results[f"{key}_recall"] = value["recall"]
            
    return results

def train(args):
    # Initialize wandb
    wandb.init(project="Polyglot-NER-project", name=args.run_name)

    # Load tokenized datasets
    print("Loading datasets...")
    tokenized_dataset = load_from_disk(os.path.join(args.data_dir, "tokenized_datasets/tokenized_dataset"))
    tokenized_hun_dataset = load_from_disk(os.path.join(args.data_dir, "tokenized_datasets/hun_tokenized_dataset"))
    tokenized_eng_dataset = load_from_disk(os.path.join(args.data_dir, "tokenized_datasets/eng_tokenized_dataset"))
    tokenized_ger_dataset = load_from_disk(os.path.join(args.data_dir, "tokenized_datasets/ger_tokenized_dataset"))

    # Mapping
    label_names = tokenized_dataset['train'].features['labels'].feature.names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    # Initialize model and tokenizer
    print(f"Initializing model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_id,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Multi-language validation sets
    eval_datasets = {
        "hun": tokenized_hun_dataset["validation"],
        "eng": tokenized_eng_dataset["validation"],
        "ger": tokenized_ger_dataset["validation"]
    }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        push_to_hub=False,
        report_to="wandb",
        run_name=args.run_name,
        fp16=torch.cuda.is_available(), # Use FP16 if GPU available
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, label_names),
        processing_class=tokenizer,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multilingual NER model.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to tokenized datasets.")
    parser.add_argument("--model_id", type=str, default="xlm-roberta-base", help="Model ID from HuggingFace Hub.")
    parser.add_argument("--output_dir", type=str, default="models/polyglot-ner-xlmr", help="Directory to save the model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--run_name", type=str, default="polyglot-ner-run-v1", help="Name for the W&B run.")
    
    args = parser.parse_args()
    train(args)
