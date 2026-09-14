import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import app, clean, match_roles, validate_profile


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_clean_function():
    text = "Python 123, Machine-Learning!"
    result = clean(text)

    assert result == "python machine learning"


def test_role_matching():
    skills = [
        "python",
        "sql",
        "pandas",
        "numpy",
        "machine learning",
    ]

    results = match_roles(skills, top_n=3)

    assert len(results) == 3
    assert "role" in results[0]
    assert "match_percent" in results[0]


def test_profile_validation_success():
    data = {
        "name": "Test Student",
        "email": "test@example.com",
        "education": "BCA",
        "skills": "python, sql, pandas",
        "experience_years": "0",
    }

    errors = validate_profile(data)

    assert errors == []


def test_profile_validation_missing_name():
    data = {
        "name": "",
        "email": "test@example.com",
        "education": "BCA",
        "skills": "python, sql",
        "experience_years": "0",
    }

    errors = validate_profile(data)

    assert "Name is required." in errors


def test_home_requires_login(client):
    response = client.get("/")

    assert response.status_code in [302, 401]


def test_available_roles_requires_login(client):
    response = client.get("/api/available-roles")

    assert response.status_code in [302, 401]
