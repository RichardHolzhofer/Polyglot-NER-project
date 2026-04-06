import argparse
import os
import sys

import wandb
from dotenv import load_dotenv

from src.config import NERConfig
from src.data_loader import NERDataLoader
from src.data_preprocessor import NERDataPreprocessor
from src.exception import NERException
from src.logger import logging
from src.trainer import PolyglotTrainer


def main():
    try:
        # Load environment variables
        load_dotenv()

        # Authenticate with Weights & Biases
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
            logging.info("Successfully logged into Weights & Biases.")
        else:
            logging.warning(
                "WANDB_API_KEY not found in environment. "
                "W&B tracking may fail or prompt for credentials."
            )

        parser = argparse.ArgumentParser(description="Polyglot NER Training Entry Point")
        parser.add_argument("--epochs", type=int, help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, help="Training batch size")
        parser.add_argument("--lr", type=float, help="Learning rate")
        parser.add_argument("--run_name", type=str, help="W&B run name")

        args = parser.parse_args()

        # Initialize Configuration
        config = NERConfig()

        # Override config with CLI arguments if provided
        if args.epochs:
            config.num_train_epochs = args.epochs
        if args.batch_size:
            config.per_device_train_batch_size = args.batch_size
        if args.lr:
            config.learning_rate = args.lr
        if args.run_name:
            config.output_model_name = args.run_name

        logging.info("--- Starting Polyglot NER Training Pipeline ---")

        # Automatic Preprocessing (it only runs if cache is missing)
        processed_full_path = os.path.join(config.data_dir, config.processed_path)
        if not os.path.exists(processed_full_path):
            logging.info(
                f"Processed dataset not found at {processed_full_path}. Running preprocessing..."
            )
            preprocessor = NERDataPreprocessor(config)
            preprocessor.run_pipeline()
        else:
            logging.info(
                f"Processed dataset found at {processed_full_path}. Skipping preprocessing step."
            )

        logging.info(f"Model ID: {config.model_id}")
        logging.info(f"Epochs:   {config.num_train_epochs}")
        logging.info(f"LR:       {config.learning_rate}")

        # Initialize Data Loader
        data_loader = NERDataLoader(config)

        # Initialize and Run Trainer
        trainer = PolyglotTrainer(config, data_loader)

        # Load and set up evaluation datasets
        dn = data_loader.load_datasets()
        eval_ds = data_loader.get_eval_datasets(dn)

        # Passing the full dictionary of evaluation datasets to maintain
        # per-language validation metrics during training.
        trainer.train(eval_dataset=eval_ds, run_name=args.run_name)

        logging.info("--- Training Pipeline Completed Successfully ---")

    except NERException as e:
        logging.error(f"Top-level NERException caught: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Untracked exception in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
