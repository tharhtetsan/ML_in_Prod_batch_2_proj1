import os
import sys
sys.path.append(os.getcwd())
from main import app  # Import your FastAPI app

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# Test Async Prediction Endpoint
@pytest.mark.asyncio
async def test_async_prediction(client):
    response = await client.post("/async")
    assert response.status_code == 200
    assert "execution_time" in response.json()
    assert response.json()["result"] == "OK"

# Test Text Generation Endpoint
@pytest.mark.asyncio
async def test_text_gen(client):
    response = await client.post("/text_gen", json={"prompt": "Generate text"})
    assert response.status_code == 200
    assert "execution_time" in response.json()
    assert isinstance(response.json()["result"], str)

# Test Audio Generation Endpoint
@pytest.mark.asyncio
async def test_audio_gen(client):
    response = await client.get("/audio_gen", params={"prompt": "Test audio"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

