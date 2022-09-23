# Bring your packages onto the path
import sys, os
import pytest
from fastapi.testclient import TestClient
sys.path.append(os.path.abspath(os.path.join('..', '')))
# Now do your import
from main import app

@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client