import pytest
from app import app


@pytest.fixture
def client():
    return app.test_client()


def test_index(client):
    assert client.get('/').status_code == 200
