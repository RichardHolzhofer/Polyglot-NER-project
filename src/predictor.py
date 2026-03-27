import os
import sys
from typing import List, Dict, Any, Optional, Union
from transformers import pipeline
from src.config import NERConfig
from src.logger import logging
from src.exception import NERException

class NERPredictor:
    """
    Production-ready predictor for the Polyglot NER model.
    Wraps the Hugging Face pipeline for easy inference.
    """
    def __init__(self, config: Optional[NERConfig] = None, model_path: Optional[str] = None):
        try:
            self.config = config or NERConfig()
            self.model_path = model_path or self.config.output_dir
            
            logging.info(f"Loading inference pipeline from: {self.model_path}")
            self.nlp = pipeline(
                "ner", 
                model=self.model_path, 
                tokenizer=self.model_path,
                aggregation_strategy="simple"
            )
            logging.info("Inference pipeline loaded successfully.")
            
        except Exception as e:
            logging.error(f"Failed to load predictor from {self.model_path}")
            raise NERException(e, sys)

    def predict(self, inputs: Union[str, List[str]]) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Predicts entities in a single string or a batch of strings.
        Native support from transformers pipeline.
        """
        try:
            return self.nlp(inputs)
        except Exception as e:
            msg = f"Prediction failed for inputs: {str(inputs)[:50]}..."
            logging.error(msg)
            raise NERException(e, sys)

