import pytest
from openai import OpenAI

from app.openai_client import create_openai_client, get_openai_model


def test_create_openai_client_successfully(monkeypatch):
    """Client should initialize when an API key is available."""

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-api-key",
    )

    client = create_openai_client()

    assert isinstance(client, OpenAI)
    assert client.api_key == "test-api-key"

    client.close()


def test_create_openai_client_without_api_key(monkeypatch, tmp_path):
    """Client creation should fail when the API key is missing."""

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    # Use an empty temporary directory without a .env file.
    monkeypatch.setattr(
        "app.openai_client.PROJECT_ROOT",
        tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match="OPENAI_API_KEY is missing",
    ):
        create_openai_client()
        
def test_create_client_loads_api_key_from_env_file(monkeypatch, tmp_path):
    """Client should load the API key from a .env file in the project root."""

    # Create a temporary .env file with the API key.
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-env-api-key\n")

    # Use the temporary directory as the project root.
    monkeypatch.setattr(
        "app.openai_client.PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )
    client = create_openai_client()

    assert isinstance(client, OpenAI)
    assert client.api_key == "test-env-api-key"

    client.close()
    
def test_get_openai_model_successfully(monkeypatch):
    """Model name should be retrieved successfully when available."""

    monkeypatch.setenv(
        "OPENAI_MODEL",
        "test-model",
    )

    model_name = get_openai_model()

    assert model_name == "test-model"
    
def test_get_openai_model_without_configuration(monkeypatch, tmp_path):
    """Missing model configuration should raise RuntimeError."""

    monkeypatch.delenv(
        "OPENAI_MODEL",
        raising=False,
    )

    # Use an empty temporary directory without a .env file.
    monkeypatch.setattr(
        "app.openai_client.PROJECT_ROOT",
        tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match="OPENAI_MODEL is missing",
    ):
        get_openai_model()
