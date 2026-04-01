import os
import sys
from typing import Dict, Optional, Union

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import wandb
from src.config import NERConfig
from src.data_loader import NERDataLoader
from src.exception import NERException
from src.logger import logging


class PolyglotTrainer:
    """
    Wrapper for the Hugging Face Trainer to handle multilingual NER training and evaluation.
    """

    def __init__(self, config: NERConfig, data_loader: NERDataLoader):
        try:
            self.config = config
            self.data_loader = data_loader
            self.metric = evaluate.load("seqeval")
            self.trainer = None
            self.model = None
        except Exception as e:
            logging.error("Failed to initialize PolyglotTrainer components.")
            raise NERException(e, sys)

    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        Internal metric computation for the Trainer.
        """
        try:
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            label_names = self.config.label_names
            true_labels = [[label_names[lb] for lb in label if lb != -100] for label in labels]
            true_predictions = [
                [label_names[p] for (p, lb) in zip(prediction, label) if lb != -100]
                for prediction, label in zip(predictions, labels)
            ]

            all_metrics = self.metric.compute(predictions=true_predictions, references=true_labels)

            results = {
                "overall_precision": all_metrics["overall_precision"],
                "overall_recall": all_metrics["overall_recall"],
                "overall_f1": all_metrics["overall_f1"],
                "overall_accuracy": all_metrics["overall_accuracy"],
            }

            for k, v in all_metrics.items():
                if k not in [
                    "overall_precision",
                    "overall_recall",
                    "overall_f1",
                    "overall_accuracy",
                ]:
                    results[f"{k}_f1"] = v["f1"]
                    results[f"{k}_precision"] = v["precision"]
                    results[f"{k}_recall"] = v["recall"]

            return results
        except Exception as e:
            logging.error("Failed to compute metrics during evaluation.")
            raise NERException(e, sys)

    def setup_trainer(self, train_dataset, eval_dataset, run_name=None):
        """
        Configures and initializes the Hugging Face Trainer.
        """
        os.makedirs(self.config.training_dir, exist_ok=True)

        if run_name:
            training_run_name = run_name
        else:
            training_run_name = self.config.output_model_name

        try:
            training_args = TrainingArguments(
                output_dir=self.config.training_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_combined_loss",
                greater_is_better=False,
                learning_rate=self.config.learning_rate,
                num_train_epochs=self.config.num_train_epochs,
                lr_scheduler_type=self.config.scheduler,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                report_to="wandb",
                bf16=True,
                bf16_full_eval=True,
                run_name=training_run_name,
                dataloader_num_workers=8,
                dataloader_persistent_workers=True,
                dataloader_prefetch_factor=2,
                push_to_hub=False,
                hub_model_id=(f"{self.config.hub_repo_id}/{training_run_name}"),
            )

            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=self.data_loader.data_collator,
                compute_metrics=self.compute_metrics,
                processing_class=self.data_loader.tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )
        except Exception as e:
            logging.error("Failed to setup Hugging Face Trainer.")
            raise NERException(e, sys)

    def train(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        run_name: Optional[str] = None,
    ):
        """
        Starts the training process with Weights & Biases tracking.
        """
        try:
            # Initialize Model
            if self.model is None:
                logging.info(f"Initializing model natively from: {self.config.model_id}")
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.config.model_id,
                    id2label=self.config.id2label,
                    label2id=self.config.label2id,
                )

            # Load and Tokenize Datasets
            logging.info("Preparing datasets for training...")
            dn = self.data_loader.load_datasets()
            train_ds = dn["gold_only"]["train"]

            # Setup Trainer
            eval_ds = eval_dataset or self.data_loader.get_eval_datasets(dn)

            self.setup_trainer(train_dataset=train_ds, eval_dataset=eval_ds, run_name=run_name)

            # Initialize W&B for training
            logging.info(f"Initializing W&B run: {run_name or self.config.output_model_name}")
            base_run_name = run_name or self.config.output_model_name
            wandb.init(project="polyglot-ner-project", name=f"{base_run_name}_training")

            # Start Training
            logging.info("Starting training loop...")
            self.trainer.train()
            # Finish Training
            wandb.finish()

            # Push best model to HuggingFace
            logging.info("Pushing model and tokenizer to Hub")
            self.trainer.push_to_hub()
            logging.info("Model pushed to Hub successfully.")

            logging.info(f"Saving finalized model and tokenizer to: {self.config.output_dir}")
            self.trainer.save_model(self.config.output_dir)
            logging.info("Model saved successfully.")

            # Clear GPU memory before testing
            if torch.cuda.is_available():
                logging.info("Clearing GPU cache before testing...")
                torch.cuda.empty_cache()

            # Initialize W&B for testing
            logging.info(f"Initializing W&B run: {run_name or self.config.output_model_name}")
            base_run_name = run_name or self.config.output_model_name
            wandb.init(project="polyglot-ner-project", name=f"{base_run_name}_test")

            # Evaluate on Test Set
            logging.info("Evaluating best model on test datasets...")
            test_sets = self.data_loader.get_test_datasets(dn)

            for test_name, test_ds in test_sets.items():
                logging.info(f"Running evaluation on test set: {test_name}")

                # Initialize Trainer for testing
                self.setup_trainer(train_dataset=None, eval_dataset=test_ds, run_name=run_name)

                test_results = self.trainer.evaluate()
                # Rename keys to include the test_name prefix for clarity in logs/W&B
                prefixed_results = {
                    f"eval/{test_name}_{k.replace('eval_', '')}": v for k, v in test_results.items()
                }
                logging.info(f"Test results for {test_name}: {prefixed_results}")
                wandb.log(prefixed_results)

            # Finish testing
            wandb.finish()

        except Exception as e:
            logging.error("An error occurred during training.")
            raise NERException(e, sys)
