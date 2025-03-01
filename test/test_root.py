import os
import sys


sys.path.append(os.getcwd())
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_read_root():
    """For health check"""
    response = client.get("/")
