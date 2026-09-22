from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

app = FastAPI(title="Todo")

STATIC = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_todos: dict[int, dict] = {}
_next_id = 1


class TodoIn(BaseModel):
    title: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/todos")
def list_todos() -> list[dict]:
    return list(_todos.values())


@app.post("/todos", status_code=201)
def create_todo(todo: TodoIn) -> dict:
    global _next_id
    item = {"id": _next_id, "title": todo.title, "done": False}
    _todos[_next_id] = item
    _next_id += 1
    return item


@app.delete("/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: int) -> None:
    if todo_id not in _todos:
        raise HTTPException(404, "not found")
    del _todos[todo_id]
