import pytest
from unittest.mock import MagicMock, patch
from src.data_loader import NERDataLoader
import os

@pytest.fixture
def mock_tokenizer():
    """Mocks a tokenizer that returns specific word_ids for testing alignment."""
    tokenizer = MagicMock()
    
    def _mock_call(tokens, **kwargs):
        class MockEncoding(dict):
            def word_ids(self, batch_index=0):
                # Corresponding to ["I", "love", "Budapest"] -> [None, 0, 1, 2, 2, None]
                return [None, 0, 1, 2, 2, None]
        
        encoding = MockEncoding({
            "input_ids": [0, 101, 102, 103, 104, 1],
            "attention_mask": [1, 1, 1, 1, 1, 1]
        })
        return encoding
    
    tokenizer.side_effect = _mock_call
    return tokenizer

def test_loader_init(config, mock_tokenizer):
    """Test standard initialization."""
    with patch("src.data_loader.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        loader = NERDataLoader(config)
        assert loader.config == config
        assert loader.tokenizer == mock_tokenizer
        assert loader.data_collator is not None

def test_tokenize_and_align_labels(config, mock_tokenizer):
    """
    VITAL TEST: Verifies that word-level labels are expanded correctly to sub-tokens.
    Input sentence: ["I", "love", "Budapest"]
    Labels: [0, 0, 5] (where 5 is B-LOC)
    Sub-tokens: [None, 'I', 'love', 'Buda', 'pest', None]
    Expected result: [-100, 0, 0, 5, -100, -100]
    """
    loader = NERDataLoader(config, tokenizer=mock_tokenizer)
    
    dummy_data = {
        "tokens": [["I", "love", "Budapest"]],
        "ner": [[0, 0, 5]]
    }
    
    result = loader.tokenize_and_align_labels(dummy_data)
    
    assert "labels" in result
    assert result["labels"][0] == [-100, 0, 0, 5, -100, -100]

def test_load_datasets_missing_file(config, mock_tokenizer):
    """Test that it raises NERException containing FileNotFoundError message if processed data is missing."""
    loader = NERDataLoader(config, tokenizer=mock_tokenizer)
    with patch("os.path.exists", return_value=False):
        from src.exception import NERException
        with pytest.raises(NERException, match="Processed dataset not found"):
            loader.load_datasets()

@patch("src.data_loader.load_from_disk")
@patch("os.path.exists")
def test_load_datasets_success(mock_exists, mock_load, config, mock_tokenizer):
    """Test successful loading and mapping logic."""
    loader = NERDataLoader(config, tokenizer=mock_tokenizer)
    
    # Mocking DatasetDict and its map function
    mock_ds = MagicMock()
    mock_ds.map.return_value = "tokenized_ds"
    mock_load.return_value = mock_ds
    
    # Ensure all path checks pass
    mock_exists.return_value = True
    
    datasets = loader.load_datasets()
    
    assert "gold_only" in datasets
    assert datasets["gold_only"] == "tokenized_ds"
    assert "hun" in datasets
    assert "ger" in datasets
    
    # Verify mapping was called
    assert mock_ds.map.call_count == 3

def test_get_eval_test_datasets(config, mock_tokenizer):
    """Test extraction of validation and test splits."""
    loader = NERDataLoader(config, tokenizer=mock_tokenizer)
    
    mock_datasets = {
        "gold_only": {"validation": "v_gold", "test": "t_gold"},
        "hun": {"validation": "v_hun", "test": "t_hun"},
        "ger": {"validation": "v_ger", "test": "t_ger"}
    }
    
    eval_sets = loader.get_eval_datasets(mock_datasets)
    assert eval_sets["combined"] == "v_gold"
    assert eval_sets["hun"] == "v_hun"
    assert eval_sets["ger"] == "v_ger"
    
    test_sets = loader.get_test_datasets(mock_datasets)
    assert test_sets["combined"] == "t_gold"
    assert test_sets["test_hun"] == "t_hun"
    assert test_sets["test_ger"] == "t_ger"
