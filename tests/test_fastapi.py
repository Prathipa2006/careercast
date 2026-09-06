"""
Milestone 4 - FastAPI Integration Tests

Tests the CareerCast FastAPI service:
1. Root endpoint
2. Health endpoint
3. Recommendation pipeline
4. Skill-gap pipeline
5. Prediction pipeline
"""

from fastapi.testclient import TestClient

from fastapi_service.main import app


client = TestClient(app)


def test_root_endpoint():
    """FastAPI root endpoint should confirm that the service is running."""
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "CareerCast API"
    assert data["status"] == "running"


def test_health_endpoint():
    """Health endpoint should return a healthy service status."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_recommendation_pipeline():
    """Recommendation endpoint should return career recommendations."""
    payload = {
        "skills": [
            "python",
            "sql",
            "pandas",
            "numpy",
            "machine learning"
        ],
        "top_k": 3
    }

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "recommendations" in data
    assert len(data["recommendations"]) == 3

    first = data["recommendations"][0]

    assert "role" in first
    assert "match_percent" in first
    assert "matched_skills" in first


def test_skill_gap_pipeline():
    """Skill-gap endpoint should calculate missing skills."""
    payload = {
        "skills": [
            "python",
            "sql",
            "pandas"
        ],
        "target_role": "Data Scientist"
    }

    response = client.post("/gap-report", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["target_role"] == "Data Scientist"
    assert "match_percent" in data
    assert "matched_skills" in data
    assert "missing_skills" in data
    assert "suggestions" in data


def test_invalid_role_returns_error():
    """Unknown target roles should return HTTP 400."""
    payload = {
        "skills": ["python", "sql"],
        "target_role": "Invalid Career Role"
    }

    response = client.post("/gap-report", json=payload)

    assert response.status_code == 400


def test_prediction_pipeline():
    """Prediction endpoint should return model predictions."""
    payload = {
        "resume_text": """
        Python developer with experience in machine learning,
        pandas, numpy, SQL, statistics and data visualization.
        """
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "model_used" in data
    assert "predictions" in data

    assert len(data["predictions"]) > 0

    first_prediction = data["predictions"][0]

    assert "category" in first_prediction
    assert "confidence" in first_prediction


def test_empty_prediction_text_is_rejected():
    """Empty resume text should return HTTP 400."""
    payload = {
        "resume_text": ""
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400