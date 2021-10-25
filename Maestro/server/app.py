import flask
import argparse
from flask import request, jsonify
import dill as pickle
import json
import torch
import numpy as np
import base64
import zlib
import datetime
from concurrent.futures import ThreadPoolExecutor


# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.utils import list_to_json, get_embedding, get_json_data
from models import load_all_applications
from Maestro.evaluator import Evaluator
from Maestro.attacker_helper.attacker_request_helper import virtual_model
# ------------------ LOCAL IMPORTS ---------------------------------

def main(applications):
    app = flask.Flask(__name__)
    app.config["DEBUG"] = True
    app.applications = applications

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
        print("Received! get_batch_output")
        json_data = request.get_json()
        application = json_data["Application_Name"]
        print("application name:", application)
        print(app.applications)
        batch_input = json_data["data"]
        labels = json_data["labels"]

        outputs = app.applications[application].get_batch_output(batch_input, labels)#.detach().cpu().numpy()
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
        if data_type == "train":
            data = app.applications[application].training_data.get_write_data()
        elif data_type == "validation":
            data = app.applications[application].validation_data.get_write_data()
        elif data_type == "test":
            data = app.applications[application].test_data.get_write_data()
        # print(data)

        json_data = get_json_data(data)
        return {"data": json_data[:5]} # {'image': [1*28*28], 'label': 7, 'uid': 0}

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

    @app.route("/attack_evaluator", methods=['POST'])
    def attack_evaluator():
        print("Evaluate the students' attack method")
        student_id = request.form["id"]
        application = request.form["Application_Name"]
        record_path = "../tmp/attack_homework/recording.txt"
        now = datetime.datetime.now()

        with open(record_path, 'a+') as f:
            f.write(str(student_id)+'\t'+ now.strftime("%Y-%m-%d %H:%M:%S")+'\t'+ str(application) +'\t')
        # record_scores(application, student_id, record_path)
        try:
            executor.submit(record_scores, application, student_id, record_path)
        except (RuntimeError, TypeError, NameError):
            print("error")

        print(student_id)
        return {"score": "server is working on it..."}

    def record_scores(student_id, application, record_path):
        print("\n working in the records")
        vm = virtual_model("http://127.0.0.1:5000", application_name="FGSM")

        evaluator = Evaluator(student_id, application, vm)
        score = evaluator.attack_evaluator()
        print("evaluator")
        print(score)
        with open(record_path, 'a+') as f:
            f.write(str(score)+'\n')
        return


    @app.route("/evaluate_result", methods=['POST'])
    def evaluate_result():
        print("check the score of the defense method")
        record_path = "../tmp/evaluator/recording.txt"

        student_id = request.form["id"]
        application = request.form["Application_Name"]
        output = []
        with open(record_path, 'r') as f:
            data = f.readlines()
            for i in data:
                print(i)
                recording = i.split('\t')
                if recording[0] == student_id:
                    output.append(recording)
        return {"score": output}

    # ------------------ END ATTACK SERVER FUNCTIONS ---------------------------


    print("Server Running...........")
    # app.run(debug=True)
    app.run(host="0.0.0.0")


if __name__ == "__main__":
    executor = ThreadPoolExecutor(20)

    parser = argparse.ArgumentParser("start the allennlp demo")
    # application_names = ["Data_Augmentation_CV"]
    application_names = ["FGSM"]
    parser.add_argument(
        "--application",
        type=str,
        action="append",
        default=application_names,
        help="if specified, only load these models",
    )
    args = parser.parse_args()

    applications = load_all_applications(args.application)
    print("All Applications Loaded.........")
    main(applications)
