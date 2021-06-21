import flask
import argparse
from flask import request, jsonify
from models import load_all_applications
import dill as pickle
import json
from Maestro.utils import list_to_json, get_embedding
import torch
import numpy as np
import base64
import zlib

def main(applications):
    app = flask.Flask(__name__)
    app.config["DEBUG"] = True
    app.applications = applications

    @app.route("/", methods=["GET"])
    def home():
        return "<h1>The Home of Maestro Server</p>"

    @app.route("/get_batch_output", methods=["POST"])
    def get_batch_output():
        print("recieved! get_batch_output")
        # print(request.form)
        img = base64.b64decode(request.form["data"].encode())
        img = zlib.decompress(img)
        img = np.frombuffer(img)
        img = img.reshape((1,1,28,28))

        application = request.form["Application_Name"]
        print("application name:", application)
        # print(app.applications)
        # uid_list = [int(x) for x in request.form.getlist("uids")]
        # print("uid_list:", uid_list)
        # print(request.files["file"])
        # pred_hook = pickle.loads(request.files["file"].read())
        # batch_input = np.fromstring(request.form["data"], dtype=float)
        batch_input = img
        labels = request.form["label"]
        # batch_input = [int(x) for x in batch_input]
        print("batch_input: ", batch_input.shape)

        outputs = app.applications[application].get_batch_output(batch_input, labels)#.detach().cpu().numpy()
        returned = list_to_json([x.cpu().detach().numpy().tolist() for x in outputs])
        return {"outputs": returned}

    @app.route("/get_batch_input_gradient", methods=["POST"])
    def get_batch_input_gradient():
        print("recieved!")
        img = base64.b64decode(request.form["data"].encode())
        img = zlib.decompress(img)
        img = np.frombuffer(img)
        img = img.reshape((1,1,28,28))

        application = request.form["Application_Name"]
        print("application name:", application)
        batch_input = img
        labels = request.form["label"]

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
        json_data = data.get_json_data()
        return {"data": json_data[0:50]} # {'image': [1*28*28], 'label': 7, 'uid': 0}

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

    print("Server Running...........")
    # app.run(debug=True)
    app.run(host="0.0.0.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("start the allennlp demo")
    # application_names = ["Universal_Attack", "FGSM", "Hotflip", "Data_Poisoning"]
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
