import pytest
from fastapi.testclient import TestClient

from server import app


def assert_response_code(response, code=200):
    if response.status_code != code:
        print(f"Expected response code {code} but found {response.status_code}. Response body: ", response.text)
    assert code == response.status_code


@pytest.fixture
def client():
    client = TestClient(app)
    yield client
