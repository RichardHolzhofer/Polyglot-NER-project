import pytest
from unittest.mock import patch
import os
from fastapi.testclient import TestClient

from app import app
from src.config import NERConfig

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables to prevent actual API interactions during testing."""
    with patch.dict(os.environ, {"WANDB_API_KEY": "fake_key", "HUGGINGFACE_HUB_TOKEN": "fake_token", "WANDB_MODE": "disabled"}):
        yield

@pytest.fixture
def mock_hf_pipeline():
    """Mocks the transformers pipeline to prevent heavy model downloads during tests."""
    def _mock_nlp(inputs, **kwargs):
        print("Mock pipeline called with:", inputs)
        # We include punctuation in the fake entity to test post-processing
        fake_entity = {"word": inputs[0:5], "entity_group": "PER", "score": 0.99, "start": 0, "end": 5}
        
        if isinstance(inputs, str):
            if not inputs:
                return []
            return [fake_entity.copy()]
        elif isinstance(inputs, list):
            return [[fake_entity.copy()] if text else [] for text in inputs]
            
    with patch("src.predictor.pipeline", return_value=_mock_nlp):
        yield _mock_nlp

@pytest.fixture
def config():
    """Provides a fresh NERConfig instance for each test."""
    return NERConfig()

@pytest.fixture
def client(mock_hf_pipeline):
    """
    Provides a FastAPI TestClient. 
    The context manager starts the lifespan event, which initializes the NERPredictor
    using our mocked hf pipeline.
    """
    with TestClient(app) as test_client:
        yield test_client
