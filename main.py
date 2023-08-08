import box
import timeit
import yaml
import uuid
import time
from dotenv import find_dotenv, load_dotenv
from src.llm import build_llm
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from threading import Thread

load_dotenv(find_dotenv())


with open("config/config.yml", "r", encoding="utf8") as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

llm = build_llm()
app = Flask(__name__)
app.config[
    "SQLALCHEMY_DATABASE_URI"
    ] = "sqlite:////Users/arhamjain/VSCodeProjects/llm/db/llm.db"
db = SQLAlchemy(app)


llm_process_map = {}


class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    uuid = db.Column(db.String, unique=True, index=True)
    prompt = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text)
    generation_time = db.Column(db.Float)
    status = db.Column(db.String, nullable=False)


@app.route("/v1/chat/completions", methods=["POST"])
def process_prompt():
    # Get the JSON request body
    data = request.get_json()

    # Check if 'prompt' key is in the request body
    if "prompt" in data:
        start = timeit.default_timer()
        result = llm(data["prompt"])
        end = timeit.default_timer()
        print(f"Time to retrieve response: {end - start}")
        return jsonify(result=result)
    else:
        return jsonify(error="No prompt provided"), 400


@app.route("/v1/chat/completions-async", methods=["POST"])
def process_prompt_async():
    data = request.get_json()

    if "prompt" in data:
        task_id = str(uuid.uuid4())
        task = Task(uuid=task_id, prompt=data["prompt"], status="incomplete")
        db.session.add(task)
        db.session.commit()

        return jsonify(uuid=task_id)
    else:
        return jsonify(error="No prompt provided"), 400


@app.route("/v1/chat/completions-async", methods=["GET"])
def get_response():
    task_uuid = request.args.get("uuid")

    if task_uuid is None:
        return jsonify(error="No UUID provided"), 400

    session = db.session
    task = session.query(Task).filter_by(uuid=task_uuid).first()
    session.close()

    if task is None:
        return jsonify(error="Task not found"), 404

    if task.status == "incomplete":
        return jsonify(status="incomplete")

    return jsonify(status=task.status, response=task.response)


def worker():
    with app.app_context():
        while True:
            print("Checking for tasks")
            session = db.session
            task = (session.query(Task)
                    .filter_by(status="incomplete").order_by(Task.id)
                    .first())

            if task is not None:
                if task.uuid in llm_process_map:
                    continue
                llm_process_map[task.uuid] = True
                run_llm(task, session)
                del llm_process_map[task.uuid]

            session.close()
            time.sleep(5)


def run_llm(task, session):
    print(f"Running LLM for task {task.uuid}")

    try:
        start = timeit.default_timer()
        result = llm(task.prompt)
        end = timeit.default_timer()
        task.generation_time = end - start
        task.response = result
        task.status = "success"
    except Exception as e:
        task.status = "failure"
        app.logger.error(f"Error running LLM: {e}")
    session.commit()


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    worker_thread = Thread(target=worker)
    worker_thread.start()
    app.run(debug=True)
    print("App started")
