def test_root_endpoint(client):
    """Test that the root endpoint returns expected welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to the Polyglot NER API!"
    docs_msg = "Visit /docs for the interactive API documentation and testing interface."
    assert data["docs"] == docs_msg
    assert data["health"] == "Visit /health to check API status."


def test_health_endpoint(client):
    """Test that health check endpoint returns healthy status and model_loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint_single_string(client):
    """Test the predict endpoint with a single string works."""
    test_sentence = {"items": "This is a single sentence test."}
    response = client.post("/predict", json=test_sentence)

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert isinstance(data["results"], list)

    entity = data["results"][0]
    expected_keys = {"word", "entity_group", "score", "start", "end"}
    assert all(k in entity for k in expected_keys)
    assert entity["word"] == "This"  # Because of data post-processing there is no space at the end


def test_predict_endpoint_batch_strings(client):
    """Test the predict endpoint with a list of strings works."""
    test_sentences = {"items": ["Sentence one.", "Sentence two."]}
    response = client.post("/predict", json=test_sentences)

    assert response.status_code == 200
    data = response.json()

    assert "results" in data.keys()
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 2
    assert isinstance(data["results"][0], list)


def test_predict_endpoint_invalid_input(client):
    """Test that FastAPI correctly rejects invalid data payloads with HTTP 422."""
    invalid_input = {"items": 12345}
    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422
    assert "detail" in response.json()
