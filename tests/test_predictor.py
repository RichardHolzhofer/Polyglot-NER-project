from unittest.mock import patch

from src.config import NERConfig
from src.predictor import NERPredictor


def test_predictor_initializes_with_hub_fallback(mock_hf_pipeline):
    """Test predictor falls back to hub model if local config.json is missing."""
    with patch("os.path.exists", return_value=False):
        config = NERConfig()
        predictor = NERPredictor(config)
        expected_path = f"{config.hub_repo_id}/{config.output_model_name}"
        assert predictor.model_path == expected_path

def test_predictor_initializes_with_local_model(mock_hf_pipeline):
    """Test predictor uses local model if config.json exists."""
    with patch("os.path.exists", return_value=True):
        config = NERConfig()
        predictor = NERPredictor(config)
        assert predictor.model_path == config.output_dir

def test_predictor_single_string_and_postprocessing(mock_hf_pipeline):
    """
    Test that prediction works for a single string.
    """
    predictor = NERPredictor(config=NERConfig())
    results = predictor.predict("-First test input sentence.")

    assert isinstance(results, list)
    assert len(results) == 1

    entity = results[0]

    assert entity["word"] == "-Firs"
    assert entity["entity_group"] == "PER"
    assert entity["score"] == 0.99
    assert entity["start"] == 0
    assert entity["end"] == 5


def test_predictor_batch_strings(mock_hf_pipeline):
    """Test that a list of strings returns a list of lists of predictions."""
    predictor = NERPredictor(NERConfig())
    results = predictor.predict(["Mondat egy.", "Mondat kettő."])

    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], list)
    assert len(results[1]) == 1
    assert results[0][0]["word"] == "Monda"
