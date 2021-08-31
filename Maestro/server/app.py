import flask
import argparse
from flask import request, jsonify
import dill as pickle
import json
import torch
import numpy as np
import base64
import zlib

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.utils import list_to_json, get_embedding, get_json_data
from models import load_all_applications
# ------------------ LOCAL IMPORTS ---------------------------------
def main(applications):
    app = flask.Flask(__name__)
    app.config["DEBUG"] = True
    app.applications = applications

    @app.route("/", methods=["GET"])
    def home():
        return "<h1>The Home of Maestro Server</p>"
    # ------------------ DEFENSE SERVER FUNCTIONS ------------------------------

    @app.route("/send_augmented_dataset", methods=["POST"])
    def send_augmented_dataset():
        print("recieved! send_augmented_dataset")

        print("Printing request")
        multi_dict = request.args
        for key in multi_dict:
            print(multi_dict.get(key))
            print(multi_dict.getlist(key))
        json_data = request.get_json()
        application = json_data["Application_Name"]
        augmented_dataset = json_data["Potato"]
        #app.applications[application].set_training_set(augmented_dataset)
        return {"result": "OK"}

    @app.route("/send_train_signal", methods=["POST"])
    def send_train_signal():
        print("recieved! send_train_signal")
        json_data = request.get_json()
        application = json_data["Application_Name"]
        app.applications[application].train()
        return {"result": "OK"}
    # ------------------ END DEFENSE SERVER FUNCTIONS --------------------------

    # ------------------ ATTACK SERVER FUNCTIONS -------------------------------

    @app.route("/get_batch_output", methods=["POST"])
    def get_batch_output():
        print("recieved! get_batch_output")
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
        print("recieved!")
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
        print("recieved!")
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
        print("recieved!")
        application = request.form["Application_Name"]
        embedding = get_embedding(app.applications[application].model).weight
        returned = list_to_json([x.detach().cpu().numpy().tolist() for x in embedding])
        return {"data": returned}

    @app.route("/convert_tokens_to_ids", methods=["POST"])
    def convert_tokens_to_ids():
        print("recieved! convert_tokens_to_ids")
        application = request.form["Application_Name"]
        tokenizer = app.applications[application].get_tokenizer()
        json_data = tokenizer.convert_tokens_to_ids(request.form["text"])
        # print(json_data)
        # print(tokenizer.convert_tokens_to_ids("a longer sentence"))
        return {"data": json_data}

    @app.route("/convert_ids_to_tokens", methods=["POST"])
    def convert_ids_to_tokens():
        print("recieved! convert_tokens_to_ids")
        application = request.form["Application_Name"]
        tokenizer = app.applications[application].get_tokenizer()
        json_data = tokenizer.convert_ids_to_tokens(int(request.form["text"]))
        # print(json_data)
        return {"data": json_data}
    # ------------------ END ATTACK SERVER FUNCTIONS ---------------------------

    print("Server Running...........")
    # app.run(debug=True)
    app.run(host="0.0.0.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("start the allennlp demo")
    application_names = ["Data_Augmentation"]

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
