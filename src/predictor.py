import os
import sys
from typing import Any, Dict, List, Optional, Union

from transformers import pipeline

from src.config import NERConfig
from src.exception import NERException
from src.logger import logging


class NERPredictor:
    """
    Production-ready predictor for the Polyglot NER model.
    Wraps the Hugging Face pipeline for easy inference.
    """

    def __init__(self, config: Optional[NERConfig] = None):
        try:
            self.config = config or NERConfig()

            # We specifically check for config.json to ensure the model is fully saved
            if os.path.exists(os.path.join(self.config.output_dir, "config.json")):
                self.model_path = self.config.output_dir
                logging.info("Using LOCAL model.")
            else:
                self.model_path = f"{self.config.hub_repo_id}/{self.config.output_model_name}"
                logging.info(
                    f"No local model found at {self.config.output_dir}. "
                    f"Falling back to HUB: {self.model_path}"
                )

            self.nlp = pipeline(
                "ner",
                model=self.model_path,
                tokenizer=self.model_path,
                aggregation_strategy="first",
            )
            logging.info("Inference pipeline loaded successfully.")

        except Exception as e:
            logging.error(f"Failed to load predictor from {self.model_path}")
            raise NERException(e, sys)

    def predict(
        self, inputs: Union[str, List[str]]
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Predicts entities in a single string or a batch of strings.
        Native support from transformers pipeline.
        """
        try:
            results = self.nlp(inputs)

            # Post-processing helper
            def process_entities(entities, text):
                for ent in entities:
                    start, end = ent["start"], ent["end"]
                    word = text[start:end]

                    # Trim leading punctuation/whitespace
                    while word and not (word[0].isalnum() or word[0] == "-"):
                        word = word[1:]
                        start += 1

                    # Trim trailing punctuation/whitespace
                    while word and not (word[-1].isalnum() or word[-1] == "-"):
                        word = word[:-1]
                        end -= 1

                    ent["start"] = start
                    ent["end"] = end
                    ent["word"] = word

            if isinstance(inputs, str):
                process_entities(results, inputs)
            elif isinstance(inputs, list):
                for text, ents in zip(inputs, results):
                    process_entities(ents, text)

            return results
        except Exception as e:
            msg = f"Prediction failed for inputs: {str(inputs)[:50]}..."
            logging.error(msg)
            raise NERException(e, sys)
