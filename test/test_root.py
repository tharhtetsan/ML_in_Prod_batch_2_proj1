import os
import sys
sys.path.append(os.getcwd())
from main import app  # Import your FastAPI app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_root():
    """ For health check """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


# Test get_student Endpoint
def test_get_student():
    response = client.post("/get_student", json={
                                                "class_name": "ML_in_Prod_1",
                                                "stu_name": "Mg ba",
                                                "stu_id": 100,
                                                "stu_age": 16
                                                })
    assert response.status_code == 200
    assert "execution_time" in response.json()
    assert response.json()["result"] == "OK"




# Test Sync Prediction Endpoint
def test_sync_prediction():
    response = client.post("/sync", params={"prompt": "test prompt"})
    assert response.status_code == 200
    assert "execution_time" in response.json()
    assert response.json()["result"] == "OK"

