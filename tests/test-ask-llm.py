import pytest
import requests
import json
from typing import Dict, List, Optional, Any

BASE_URL = "http://localhost:8000"  # Change this to your actual API URL

def test_api_with_model_type():
    """Test using just model_type parameter."""
    payload = {
        "prompt": "What is Python?",
        "model_type": "default"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-3-7-sonnet-latest"
    assert data["use_thinking"] is False

def test_api_with_provider_and_model():
    """Test using provider and model_name parameters."""
    payload = {
        "prompt": "What is Python?",
        "provider": "anthropic",
        "model_name": "claude-3-7-sonnet-latest"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-3-7-sonnet-latest"

def test_api_with_system_prompt():
    """Test using both system_prompt and prompt."""
    payload = {
        "prompt": "What is Python?",
        "system_prompt": "You are a programming expert. Keep answers brief and technical.",
        "model_type": "default"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # The response should be technical and brief, but that's hard to test programmatically

def test_api_with_xml_tags():
    """Test using just xml_tags."""
    payload = {
        "prompt": "Compare Python and JavaScript",
        "model_type": "default",
        "xml_tags": ["advantages", "disadvantages", "conclusion"]
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "parsed_response" in data
    assert "advantages" in data["parsed_response"]
    assert "disadvantages" in data["parsed_response"]
    assert "conclusion" in data["parsed_response"]

def test_api_with_xml_outer_tag():
    """Test using just xml_outer_tag."""
    payload = {
        "prompt": "What is Python?",
        "model_type": "default",
        "xml_outer_tag": "response"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "parsed_response" in data
    # Since we didn't specify xml_tags, parsed_response should be empty
    assert data["parsed_response"] == {}

def test_api_with_xml_outer_and_inner_tags():
    """Test using both xml_outer_tag and xml_tags."""
    payload = {
        "prompt": "Compare Python and JavaScript",
        "model_type": "default",
        "xml_outer_tag": "comparison",
        "xml_tags": ["python", "javascript", "conclusion"]
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "parsed_response" in data
    assert "python" in data["parsed_response"]
    assert "javascript" in data["parsed_response"]
    assert "conclusion" in data["parsed_response"]

def test_api_with_thinking_model():
    """Test using a thinking model."""
    payload = {
        "prompt": "What is Python?",
        "model_type": "default-thinking"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "thinking" in data
    assert data["thinking"] != ""
    assert data["use_thinking"] is True
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-3-7-sonnet-20250219"

def test_api_with_fast_model():
    """Test using a fast model."""
    payload = {
        "prompt": "What is Python?",
        "model_type": "fast"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "groq"
    assert data["model"] == "llama-3.3-70b-versatile"

def test_api_with_cheap_model():
    """Test using a cheap model."""
    payload = {
        "prompt": "What is Python?",
        "model_type": "cheap"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "google"
    assert data["model"] == "gemini-2.0-flash"

def test_api_with_invalid_model_type():
    """Test with an invalid model_type."""
    payload = {
        "prompt": "What is Python?",
        "model_type": "invalid-model-type"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Invalid model type" in data["detail"]

def test_api_with_invalid_provider():
    """Test with an invalid provider."""
    payload = {
        "prompt": "What is Python?",
        "provider": "invalid-provider",
        "model_name": "some-model"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Invalid model" in data["detail"]

def test_api_with_max_tokens():
    """Test with custom max_tokens."""
    payload = {
        "prompt": "Write a long essay about Python",
        "model_type": "default",
        "max_tokens": 100  # Limit to a short response
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Check that the response is relatively short
    # This is an approximate check since token count != character count
    assert len(data["response"]) < 800  # ~100 tokens ≈ 400-800 characters

def test_api_minimal_request():
    """Test with just the required prompt parameter."""
    payload = {
        "prompt": "What is Python?"
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # Should use default model
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-3-7-sonnet-latest"



def test_fast_model_with_thinking():
    """Test with a fast model and thinking."""
    payload = {
        "prompt": "What is Python? Return your response in <python_pros> , <python_cons>, <js_pros>, <js_cons> and <recommendation> tags.",
        "model_type": "fast-thinking",
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "thinking" in data
    assert data["thinking"] != ""


def test_api_complex_request():
    """Test with a complex combination of parameters."""
    payload = {
        "prompt": "Compare Python and JavaScript for web development. Return your response in <python_pros> , <python_cons>, <js_pros>, <js_cons> and <recommendation> tags.",
        "system_prompt": "You are a senior web developer with 10 years of experience.",
        "model_type": "default-thinking",
        "max_tokens": 2500,
        "xml_outer_tag": "comparison",
        "xml_tags": ["python_pros", "python_cons", "js_pros", "js_cons", "recommendation"]
    }
    response = requests.post(f"{BASE_URL}/v1/llm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "thinking" in data
    assert data["thinking"] != ""
    assert "parsed_response" in data
    assert len(data["parsed_response"]) == 5  # All 5 tags should be present
    assert "python_pros" in data["parsed_response"]
    assert "recommendation" in data["parsed_response"]

if __name__ == "__main__":
    # This allows running the tests directly with python test_llm_api.py
    pytest.main(["-xvs", __file__])