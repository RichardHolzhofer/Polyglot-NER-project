import os
import sys
from typing import Dict

from datasets import (
    DatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import AutoTokenizer

from src.config import NERConfig
from src.exception import NERException
from src.logger import logging


class NERDataPreprocessor:
    """
    Handles the transformation of raw datasets into harmonized, BIO-tagged formats.
    """
    def __init__(self, config: NERConfig):
        try:
            self.config = config
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)

        except Exception as e:
            logging.error("Failed to initialize NERDataPreprocessor components.")
            raise NERException(e, sys)

    def load_raw(self) -> Dict[str, DatasetDict]:
        """
        Loads raw datasets, checking local cache first, then downloading if missing.
        """
        try:
            datasets = {}
            sources = {
                "hun": (
                    self.config.hun_raw_id,
                    self.config.hun_raw_path,
                    {"revision": "convert/parquet"}
                ),
                "ger": (
                    self.config.ger_raw_id,
                    self.config.ger_raw_path,
                    {"name": self.config.ger_raw_subset}
                )
            }

            for key, (hf_id, local_path, hf_kwargs) in sources.items():
                if os.path.exists(local_path):
                    logging.info(
                        f"Loading {key} raw dataset from local cache: {local_path}"
                    )
                    datasets[key] = load_from_disk(local_path)
                else:
                    logging.info(
                        f"Downloading {key} raw dataset from Hugging Face Hub: {hf_id}"
                    )
                    ds = load_dataset(hf_id, **hf_kwargs)
                    datasets[key] = ds

                    # Saving datasets locally for future use and DVC tracking
                    logging.info(f"Creating folder for {key} dataset")
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    ds.save_to_disk(local_path)
                    logging.info(f"Raw {key} dataset saved to: {local_path}")

            return datasets

        except Exception as e:
            logging.error("Failed to download raw datasets.")
            raise NERException(e, sys)

    def harmonize_hun(self, ds: DatasetDict) -> DatasetDict:
        """Hungarian is already in master format, just ensuring consistent split names."""
        try:
            logging.info("Harmonizing Hungarian dataset...")

            if "ner_tags" in ds["train"].column_names:
                ds = ds.rename_column("ner_tags", "ner")
            ds = ds.select_columns(["tokens", "ner"])

            if self.config.master_dataset == "hun":
                self.config.master_features = ds["train"].features

            # --- FIX: Just return the original splits ---
            return ds
        except Exception as e:
            logging.error("Failed to harmonize Hungarian dataset.")
            raise NERException(e, sys)

    def harmonize_ger(self, ds: DatasetDict) -> DatasetDict:
        """Maps German BIO tags to the Master Hungarian schema."""
        try:
            logging.info("Harmonizing German dataset...")
            ds = ds.rename_column("ner_tags", "ner")

            # This is good! Safely renaming 'dev' to 'validation'
            if "dev" in ds:
                ds["validation"] = ds.pop("dev")
            ger_labels = ds['train'].features['ner'].feature.names
            ger_id2label = {id: name for id, name in enumerate(ger_labels)}
            master_label2id = self.config.label2id

            def map_ids(batch):
                # Using the safe .get() fallback from earlier!
                batch["ner"] = [
                    [
                        master_label2id.get(ger_id2label[tag_id], master_label2id["O"])
                        for tag_id in row
                    ]
                    for row in batch["ner"]
                ]
                return batch

            ds = ds.map(map_ids, batched=True)
            ds = ds.select_columns(["tokens", "ner"])

            if self.config.master_dataset == "ger":
                self.config.master_features = ds["train"].features
            # --- FIX: Just return the canonical splits ---
            return ds

        except Exception as e:
            logging.error("Failed to harmonize German dataset.")
            raise NERException(e, sys)

    def cast_master_dataset_schema(self, hun_ds: DatasetDict, ger_ds: DatasetDict):
        """Cast master dataset schema for enabling interleaving datasets."""
        try:
            # We MUST select matching columns first, otherwise the cast will fail
            columns_to_keep = list(self.config.master_features.keys())
            hun_ds = hun_ds.select_columns(columns_to_keep)
            ger_ds = ger_ds.select_columns(columns_to_keep)

            hun_ds = hun_ds.cast(self.config.master_features)
            ger_ds = ger_ds.cast(self.config.master_features)

            return hun_ds, ger_ds

        except Exception as e:
            logging.error("Failed to cast master dataset schema.")
            raise NERException(e, sys)

    def run_pipeline(self):
        """Orchestrates the full preprocessing and saves to disk."""
        try:
            raw = self.load_raw()

            processed_hun = self.harmonize_hun(raw["hun"])
            processed_ger = self.harmonize_ger(raw["ger"])

            # Cast master dataset schema
            processed_hun, processed_ger = self.cast_master_dataset_schema(
                processed_hun,
                processed_ger
            )

            # Save individual processed sets
            base_out = self.config.data_dir
            for name, ds in [("hun", processed_hun), ("ger", processed_ger)]:
                config_path = getattr(self.config, f"{name}_processed_path")
                path = os.path.join(base_out, config_path)
                ds.save_to_disk(path)
                logging.info(f"Saved processed {name} dataset to {path}")

            # Create the Gold-Only (Hun + Ger) combined training set
            # We use interleave_datasets to ensure balanced language representation
            gold_train = interleave_datasets(
                [processed_hun["train"], processed_ger["train"]],
                seed=42,
                stopping_strategy="all_exhausted"  # Ensures no data is lost
            )

            # Validation and test sets can be concatenated
            gold_val = concatenate_datasets(
                [processed_hun["validation"], processed_ger["validation"]]
            )
            gold_test = concatenate_datasets(
                [processed_hun["test"], processed_ger["test"]]
            )

            gold_ds = DatasetDict({
                'train': gold_train,
                'validation': gold_val,
                'test': gold_test
            })
            gold_path = os.path.join(base_out, self.config.processed_path)
            gold_ds.save_to_disk(gold_path)
            logging.info(f"Saved combined Gold-Only dataset to {gold_path}")

        except Exception as e:
            logging.error("Data pipeline execution failed.")
            raise NERException(e, sys)
