import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from datasets import Features


@dataclass
class NERConfig:
    # Model config
    model_id: str = "xlm-roberta-base"

    # Data config
    data_dir: str = "data"


    # Local Raw Data Paths
    raw_data_dir: str = "data/raw"
    hun_raw_path: str = "data/raw/hun_raw"
    ger_raw_path: str = "data/raw/ger_raw"

    # Raw Data Sources
    hun_raw_id: str = "ficsort/SzegedNER"
    ger_raw_id: str = "bltlab/open-ner-core-types"
    ger_raw_subset: str = "GermEval_deu"

    # Processed Data Paths
    processed_path: str = "processed/gold_only_processed"
    hun_processed_path: str = "processed/hun_processed"
    ger_processed_path: str = "processed/ger_processed"

    # Training hyperparameters
    learning_rate: float = 1e-5
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 96
    per_device_eval_batch_size: int = 192
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"

    # Label config
    label_names: List[str] = field(default_factory=lambda: [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ])

    # Output config
    training_dir: str = "model_checkpoints/"
    output_dir: str = "final_model"
    output_model_name:str = "xlm-roberta-ner-hun-ger"
    hub_repo_id: str = field(default_factory=lambda: os.getenv("HUB_REPO_ID"))

    # Master features
    master_dataset: str = "hun"
    master_features: Optional[Features] = None

    # Mappings
    @property
    def id2label(self) -> Dict[int, str]:
        return {i: label for i, label in enumerate(self.label_names)}

    @property
    def label2id(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.label_names)}
