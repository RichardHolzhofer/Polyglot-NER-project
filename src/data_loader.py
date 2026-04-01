import os
import sys
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from src.config import NERConfig
from src.exception import NERException
from src.logger import logging


class NERDataLoader:
    """
    Handles loading and on-the-fly tokenization of processed datasets.
    """

    def __init__(self, config: NERConfig, tokenizer: Optional[AutoTokenizer] = None):
        try:
            self.config = config
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model_id)
            self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        except Exception as e:
            logging.error("Failed to initialize NERDataLoader components.")
            raise NERException(e, sys)

    def load_datasets(self) -> Dict[str, DatasetDict]:
        """
        Loads the processed datasets and tokenizes them on-the-fly.
        """
        try:
            logging.info("Loading and tokenizing datasets...")

            # 1. Load the primary gold-only dataset
            gold_path = os.path.join(self.config.data_dir, self.config.processed_path)
            if not os.path.exists(gold_path):
                raise FileNotFoundError(
                    f"Processed dataset not found: {gold_path}. Run data_preprocessing.py first."
                )

            gold_ds = load_from_disk(gold_path)
            tokenized_gold = gold_ds.map(self.tokenize_and_align_labels, batched=True)

            datasets = {"gold_only": tokenized_gold}

            # 2. Load individual languages for evaluation
            for lang, path_attr in [("hun", "hun_processed_path"), ("ger", "ger_processed_path")]:
                lang_path = os.path.join(self.config.data_dir, getattr(self.config, path_attr))
                if os.path.exists(lang_path):
                    lang_ds = load_from_disk(lang_path)
                    datasets[lang] = lang_ds.map(self.tokenize_and_align_labels, batched=True)
                    logging.info(f"Loaded and tokenized {lang} dataset.")

            return datasets

        except Exception as e:
            logging.error("Failed to load/tokenize datasets.")
            raise NERException(e, sys)

    def tokenize_and_align_labels(self, ds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aligns word-level BIO tags with the current tokenizer's sub-tokenization.
        Uses the 'first-subtoken only' approach (-100 for follow-ups).
        """
        try:
            tokenized_inputs = self.tokenizer(
                ds["tokens"], truncation=True, is_split_into_words=True
            )

            labels = []
            for i, label in enumerate(ds["ner"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        # Special tokens like <s> or </s>
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        # This is the FIRST sub-token of a new word
                        label_ids.append(label[word_idx])
                    else:
                        # This is a follow-up sub-token of the same word
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        except Exception as e:
            logging.error("Failed to tokenize and align labels.")
            raise NERException(e, sys)

    def get_eval_datasets(self, tokenized_datasets: Dict[str, DatasetDict]) -> Dict[str, Dataset]:
        """
        Returns a dictionary of tokenized validation sets for multi-language evaluation.
        """
        try:
            eval_sets = {"combined": tokenized_datasets["gold_only"]["validation"]}

            for lang in ["hun", "ger"]:
                if lang in tokenized_datasets:
                    eval_sets[lang] = tokenized_datasets[lang]["validation"]

            return eval_sets
        except Exception as e:
            logging.error("Failed to get evaluation datasets.")
            raise NERException(e, sys)

    def get_test_datasets(self, tokenized_datasets: Dict[str, DatasetDict]) -> Dict[str, Dataset]:
        """
        Returns a dictionary of tokenized test sets.
        """
        try:
            test_sets = {"combined": tokenized_datasets["gold_only"]["test"]}

            for lang in ["hun", "ger"]:
                if lang in tokenized_datasets:
                    test_sets[f"test_{lang}"] = tokenized_datasets[lang]["test"]

            return test_sets
        except Exception as e:
            logging.error("Failed to get test datasets.")
            raise NERException(e, sys)
