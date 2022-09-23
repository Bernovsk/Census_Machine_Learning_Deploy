# Bring your packages onto the path
import sys, os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('main'), '..')))
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client