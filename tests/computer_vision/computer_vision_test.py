from pathlib import Path
from fastapi.testclient import TestClient

from utils.helper import find_project_root

def test_computer_cision_api():
    from computer_vision.api import app

    client = TestClient(app.app)
    test_image_path = find_project_root() / Path("datasets/computer_vision/Emotions/Angry/10148.png")
    response = client.post("/predict", files={"file": open(test_image_path, "rb")})

    assert response.status_code == 200
    data = response.json()
    assert "class_id" in data
    assert "class_name" in data
    assert "confidence" in data
    assert isinstance(data["class_id"], int)
    assert isinstance(data["class_name"], str)
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0

if __name__ == "__main__":
    test_computer_cision_api()