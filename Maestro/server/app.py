import flask
import argparse
from flask import request, jsonify
from models import load_all_applications
import dill as pickle
import json
from Maestro.utils import make_json
import torch


def main(applications):
    app = flask.Flask(__name__)
    app.config["DEBUG"] = True
    app.applications = applications

    @app.route("/", methods=["GET"])
    def home():
        return "<h1>The Home of Maestro Server</p>"

    @app.route("/get_batch_output", methods=["POST"])
    def get_batch_output():
        print("recieved!")
        # print(request)
        # print(request.form)
        # print(request.get_data())
        # print(request.files)
        # print(request.json)

        uid_list = [int(x) for x in request.form.getlist("uids")]
        print(uid_list)
        application = request.form["Application_Name"]
        print(application)
        # print(request.files["file"])
        pred_hook = pickle.loads(request.files["file"].read())
        # print(pred_hook(1))
        outputs = app.applications[application].get_batch_output(
            uid_list, request.form["data_type"], pred_hook
        )
        print(outputs)

        return {"outputs": make_json(list(outputs))}

    @app.route("/get_data", methods=["POST"])
    def get_data():
        print("recieved!")
        application = request.form["Application_Name"]
        data_type = request.form["data_type"]
        if data_type == "train":
            data = app.applications[application].training_data.get_write_data()
        elif data_type == "validation":
            data = app.applications[application].validation_data.get_write_data()
        json_data = data.get_json_data()
        return {"data": json_data}

    @app.route("/convert_tokens_to_ids", methods=["POST"])
    def convert_tokens_to_ids():
        print("recieved! convert_tokens_to_ids")
        application = request.form["Application_Name"]
        tokenizer = app.applications[application].get_tokenizer()
        json_data = tokenizer.convert_tokens_to_ids("the")
        print(json_data)
        print(tokenizer.convert_tokens_to_ids("a longer sentence"))
        return {"data": json_data}

    print("server running")
    app.run(debug=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("start the allennlp demo")
    parser.add_argument(
        "--application",
        type=str,
        action="append",
        default=["Universal_Attack", "FGSM", "Hotflip"],
        help="if specified, only load these models",
    )
    args = parser.parse_args()
    applications = load_all_applications(args.application)

    main(applications)
