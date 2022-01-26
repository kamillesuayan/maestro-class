import flask
import argparse
from flask import request, jsonify
import dill as pickle
import json
import torch
import numpy as np
import time
import base64
import zlib
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from celery import Celery
import os

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.utils import list_to_json, get_embedding, get_json_data
from models import load_all_applications
from Maestro.evaluator import Evaluator
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from Maestro.Attack_Defend.Perturb_Transform import perturb_transform

# ------------------ LOCAL IMPORTS ---------------------------------

# executor = ThreadPoolExecutor(1)
# application_config_file = "Server_Config/Genetic_Attack.json"
# application_config_file = "Server_Config/FGSM_Attack.json"
application_config_file = "Server_Config/Attack_Project.json"
# application_config_file = "Server_Config/Adv_Training.json"
# application_config_file = "Server_Config/Defense_Project.json"

server_config_file = "Server_Config/Server.json"
with open(server_config_file,"r") as f:
    server_configs = json.load(f)
IP_ADDR = server_configs["ip"]
PORT = server_configs["port"]
TASK_QUEUE = server_configs["task_queue"]

applications,application_configs,attacker_path_list = load_all_applications(application_config_file)
print("All Applications Loaded.........")

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config.update(
    CELERY_BROKER_URL='redis://127.0.0.1:6379/0',
    CELERY_RESULT_BACKEND='redis://127.0.0.1:6379/0'
)
app.applications = applications
################################# MAKE TASK QUEUE WITH CELERY ####################################################
def make_celery(app):
    celery = Celery(
        app.name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
celery = make_celery(app)
@celery.task()
def wait(a):
    # for testing purpose
    time.sleep(a)
    print("finish sleeping")
    return
@celery.task()
def append_to_queue(student_id, application, record_path, task):
    # print("processing!")
    # time.sleep(5)
    # print("finsh!")
    print("Appending to queue!")
    score = record_scores(student_id, student_name, application, record_path, task)
    # try:
    #     thread_temp = executor.submit(
    #         record_scores, student_id, application, record_path, task
    #     )
    #     print(thread_temp.result())  # multithread debugging: print errors
    # except BaseException as error:
    #     print("An exception occurred: {}".format(error))
    return score
    # return thread_temp.result()
################################# MAKE TASK QUEUE WITH CELERY ####################################################
@celery.task()
def record_scores(student_id, student_name, application, record_path, task):
    global applications
    print("\nworking in the records: ", task, application)
    # if task == "defense_project":
    #     evaluator = Evaluator(application, student_id, None, task)
    #     score = evaluator.defense_evaluator_project()
    # else:
    vm = virtual_model(
        "http://"+ IP_ADDR + ":" + PORT, application_name=application
    )  # "FGSM"

    application_idx = 0
    for i in range(len(application_configs["Application"])):
        if application_configs["Application"][i]["name"] == application:
            application_idx = i
    model_name = application_configs["Application"][application_idx]["model"]["name"]
    evaluator = Evaluator(
        application,
        model_name,
        student_id,
        student_name,
        vm,
        task,
        app_pipeline=app.applications[application],
    )
    print(f"the task is {task}")
    if task == "attack_homework":
        all_metrics = []
        for i in range(1):
            metrics = evaluator.attack_evaluator()
            # metrics = {}
            # metrics['score'] = 0
            all_metrics.append(metrics)
    elif task == "attack_project":
        all_metrics = []
        for i in range(1):
            metrics = evaluator.attack_evaluator_project()
            all_metrics.append(metrics)
    elif task == "defense_homework":
        metrics = evaluator.defense_evaluator(IP_ADDR, PORT, model_name, app.applications, attacker_path_list)
    elif task == "defense_project":
        score = evaluator.defense_evaluator_project()
    elif task == "war_attack":
        score = evaluator.attack_evaluator()
    elif task == "war_defend":
        score = evaluator.defense_evaluator_war(app.applications,attacker_path_list)
    else:
        print("loading evaulator error")

    print("evaluator")
    # grade score is for attack hw only
    #grade_score = metrics["grade_score"]
    leaderboard_score = metrics["leaderboard_score"]
    #print(grade_score, record_path)
    print(record_path)
    print(leaderboard_score, record_path)
    with open(record_path, "a+") as f:
        #f.write(str(grade_score) + " " + str(leaderboard_score) + "\n")
        f.write(str(leaderboard_score) + "\n")
    return metrics

def add_to_app(name, pipeline):
    global applications
    applications[name] = pipeline

def main():
    @app.route("/", methods=["GET"])
    def home():
        return "<h1>The Home of Maestro Server</p>"

    # ------------------ AUGMENTED DATA SERVER FUNCTIONS ------------------------------

    @app.route("/send_augmented_dataset", methods=["POST"])
    def send_augmented_dataset():
        print(app.applications)
        print("Received! send_augmented_dataset")
        json_data = request.get_json()
        application = json_data["Application_Name"]
        augmented_dataset = json_data["data"]
        app.applications[application].set_training_set(augmented_dataset)
        return {"Done": "OK"}

    @app.route("/send_train_signal", methods=["POST"])
    def send_train_signal():
        print("Received! send_train_signal")
        json_data = request.get_json()
        application = json_data["Application_Name"]
        app.applications[application].train()
        return {"Done": "OK"}

    # ------------------ END AUGMENTED DATA SERVER FUNCTIONS --------------------------

    @app.route("/send_loss_function", methods=["POST"])
    def send_detector_model():
        print("Received! send_loss_function")
        json_data = request.get_json()
        application = json_data["Application_Name"]
        model_dict = json_data["model"]
        model = model.load_state_dict(model_dict)
        app.applications[application].set_loss_function(model)
        metrics = app.applications[application].detection_test()
        return {"Done": metrics}

    # ------------------ ATTACK SERVER FUNCTIONS -------------------------------

    @app.route("/get_batch_output", methods=["POST"])
    def get_batch_output():
        # print("Received! get_batch_output")
        json_data = request.get_json()
        application = json_data["Application_Name"]
        # print("application name:", application)
        # print(app.applications)
        batch_input = json_data["data"]
        labels = json_data["labels"]
        # print("flag\n:")
        # print(app.applications["temp_war_defense_eval"])
        print(applications.keys())
        outputs = applications[application].get_batch_output(
            batch_input, labels
        )  # .detach().cpu().numpy()
        returned = list_to_json([x.cpu().detach().numpy().tolist() for x in outputs])
        return {"outputs": returned}

    @app.route("/get_batch_input_gradient", methods=["POST"])
    def get_batch_input_gradient():
        print("Received!")
        json_data = request.get_json()
        application = json_data["Application_Name"]
        batch_input = json_data["data"]
        labels = json_data["labels"]
        outputs = app.applications[application].get_batch_input_gradient(
            batch_input, labels
        )
        returned = list_to_json([x.cpu().numpy().tolist() for x in outputs])
        return {"outputs": returned}

    @app.route("/get_data", methods=["POST"])
    def get_data():
        print("Received!")
        application = request.form["Application_Name"]
        data_type = request.form["data_type"]
        # print(f"perturb: {request.form['perturbation']}")
        if data_type == "train":
            data = app.applications[application].training_data.get_write_data()
        elif data_type == "validation":
            data = app.applications[application].validation_data.get_write_data()
        elif data_type == "test":
            data = app.applications[application].test_data.get_write_data()
        # print(data)
        # if request.form["perturbation"] != "":
        #     data = perturb_transform(
        #         app.applications[application], data, request.form["perturbation"]
        #     )
        # json_data = get_json_data(data)
        # this won't work right now, cause someone change code in processing_data.py
        # print(data)
        json_data = data.get_json_data()
        return {"data": json_data}  # {'image': [1*28*28], 'label': 7, 'uid': 0}

    @app.route("/switch_weights", methods=["POST"])
    def switch_weights():
        print("Received!")
        application = request.form["Application_Name"]
        file_path = request.form["file_path"]
        app.applications[application].model.load_state_dict(
            torch.load(file_path, map_location="cpu")
        )
        print("success")
        return "success"

    ##
    @app.route("/get_model_embedding", methods=["POST"])
    def get_model_embedding():
        print("Received!")
        application = request.form["Application_Name"]
        embedding = get_embedding(app.applications[application].model).weight
        returned = list_to_json([x.detach().cpu().numpy().tolist() for x in embedding])
        return {"data": returned}

    @app.route("/convert_tokens_to_ids", methods=["POST"])
    def convert_tokens_to_ids():
        print("Received! convert_tokens_to_ids")
        application = request.form["Application_Name"]
        tokenizer = app.applications[application].get_tokenizer()
        json_data = tokenizer.convert_tokens_to_ids(request.form["text"])
        # print(json_data)
        # print(tokenizer.convert_tokens_to_ids("a longer sentence"))
        return {"data": json_data}

    @app.route("/convert_ids_to_tokens", methods=["POST"])
    def convert_ids_to_tokens():
        print("Received! convert_tokens_to_ids")
        application = request.form["Application_Name"]
        tokenizer = app.applications[application].get_tokenizer()
        json_data = tokenizer.convert_ids_to_tokens(int(request.form["text"]))
        # print(json_data)
        return {"data": json_data}

    @app.route("/file_evaluator", methods=["POST"])
    def file_evaluator():
        print("Evaluate the students' method.")
        student_id = request.form.get("id")
        if student_id == None:
            student_id = ""
        else:
            student_id = str(student_id) + "-"
        student_name = request.form["student_name"]

        print("Student id", student_id)
        application = request.form["Application_Name"]
        task = request.form["task"]
        try:
            submission = request.files["solution"]
        except:
            submission = None
        if submission:
            filename = str(application) + "_" + str(student_name) + ".py"
            print(submission)
            submission.save(os.path.join("../../playground/" + str(task) + "/", filename))

        # record_path = Path("../tmp/" + task + "/recording.txt")
        record_path = Path("../../playground/" + task + "/recording_"+str(student_id)+str(student_name)+".txt")

        record_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        with open(record_path, "a+") as f:
            f.write('\n' +
                #str(student_id)+"-"+str(student_name)
                str(student_name)
                + "\t"
                + now.strftime("%Y-%m-%d %H:%M:%S")
                + "\t"
                + str(application)
                + "\t"
            )

        # uid = int(os.environ.get('SUDO_UID'))
        # gid = int(os.environ.get('SUDO_GID'))
        # os.chown(record_path, uid, gid)
        # record_scores(application, student_id, record_path)
        # record_scores(application, student_id, record_path)
        # print(record_path,str(record_path),str(record_path.stem))
        job = (student_id, student_name, application, str(record_path), task)
        if TASK_QUEUE:
            append_to_queue.delay(student_id, student_name, application, str(record_path), task)
            # append_to_queue.apply(args=job)
            # wait.apply(3)
        else:
            # try:
            record_scores(student_id, student_name, application, record_path, task)

            # thread_temp = executor.submit(
            #     record_scores, student_id, application, record_path, task)
            # print(thread_temp.result())  # multithread debugging: print errors
            # except BaseException as error:
            #     print("An exception occurred: {}".format(error))
        return {"feedback": "server is working on it..."}

    @app.route("/retrieve_result", methods=["POST"])
    def retrieve_result():
        print("check the score of the defense method")
        task = request.form["task"]
        student_id = request.form.get("id")
        if student_id == None:
            student_id = ""
        else:
            student_id = str(student_id) + "-"
        student_name = request.form["name"]
        record_path = "../../playground/" + str(task) + "/recording_"+str(student_id)+str(student_name)+".txt"

        application = request.form["Application_Name"]
        output = []
        print(record_path)
        if not os.path.exists(record_path):
            return {"score": "No result!"}
        with open(record_path, "r") as f:
            data = f.readlines()
            for i in data:
                print(i)
                recording = i.split("\t")
                if recording[0] == student_name:
                    output.append(recording)
        return {"score": output}

    @app.route("/evaluate_result", methods=["POST"])
    def evaluate_result():
        print("check the score of the defense method")
        task = request.form["task"]
        student_id = request.form.get("id")
        if student_id == None:
            student_id = ""
        else:
            student_id = str(student_id) + "-"
        student_name = request.form["name"]
        record_path = "../../playground/" + str(task) + "/recording_"+str(student_id)+str(student_name)+".txt"

        application = request.form["Application_Name"]
        output = []
        print(record_path)
        if not os.path.exists(record_path):
            return {"score": "No result!"}
        with open(record_path, "r") as f:
            data = f.readlines()
            for i in data:
                print(i)
                recording = i.split("\t")
                if recording[0] == student_id:
                    output.append(recording)
        json_score = {"score": output[-1][3].strip(), "output": output[-1][1], "leaderboard": [{"name": "Score", "value": output[-1][3].strip()}] }
        return json_score

    # ------------------ END ATTACK SERVER FUNCTIONS ---------------------------

    print("Server Running...........")
    # app.run(debug=True)
    # app.run(host="0.0.0.0", port=443)
    app.run(host=IP_ADDR, port=PORT)



if __name__ == "__main__":
    main()
