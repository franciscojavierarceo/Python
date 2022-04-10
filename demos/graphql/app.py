import os
from api import app, db
from datetime import datetime
from api.models import Todo

from ariadne import (
    load_schema_from_path,
    make_executable_schema,
    graphql_sync,
    snake_case_fallback_resolvers,
    ObjectType,
)
from ariadne.constants import PLAYGROUND_HTML

from flask import request, jsonify
from api.queries import resolve_todos, resolve_todo
from api.mutations import (
    resolve_create_todo,
    resolve_mark_done,
    resolve_delete_todo,
    resolve_update_due_date,
)

query = ObjectType("Query")

query.set_field("todos", resolve_todos)
query.set_field("todo", resolve_todo)

mutation = ObjectType("Mutation")
mutation.set_field("createTodo", resolve_create_todo)
mutation.set_field("markDone", resolve_mark_done)
mutation.set_field("deleteTodo", resolve_delete_todo)
mutation.set_field("updateDueDate", resolve_update_due_date)

type_defs = load_schema_from_path("schema.graphql")
schema = make_executable_schema(
    type_defs, query, mutation, snake_case_fallback_resolvers
)


@app.route("/graphql", methods=["GET"])
def graphql_playground():
    return PLAYGROUND_HTML, 200


@app.route("/graphql", methods=["POST"])
def graphql_server():
    data = request.get_json()

    success, result = graphql_sync(schema, data, context_value=request, debug=app.debug)

    status_code = 200 if success else 400
    return jsonify(result), status_code


def insert_data() -> None:
    if len(db.session.query(Todo).all()) == 0:
        db.create_all()
        today = datetime.today().date()
        todo = Todo(description="Run a marathon", due_date=today, completed=False)
        todo.to_dict()
        db.session.add(todo)
        db.session.commit()
        print("table created")


def main():
    insert_data()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
