from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_create_list_delete():
    r = client.post("/todos", json={"title": "write tests"})
    assert r.status_code == 201
    todo = r.json()
    assert todo["title"] == "write tests" and todo["done"] is False
    assert any(t["id"] == todo["id"] for t in client.get("/todos").json())
    assert client.delete(f"/todos/{todo['id']}").status_code == 204
    assert client.delete(f"/todos/{todo['id']}").status_code == 404
