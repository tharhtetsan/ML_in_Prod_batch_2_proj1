import os
import sys
#sys.path.append(os.getcwd())
from main import app  # Import your FastAPI app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_root():
    """ For health check """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
