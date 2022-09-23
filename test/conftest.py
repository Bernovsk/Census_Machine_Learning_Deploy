import pytest
from fastapi import TestClient
from main import app

@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client