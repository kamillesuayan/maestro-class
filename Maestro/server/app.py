import flask
import argparse
from flask import request, jsonify
from models import load_all_applications


def main(applications):
    app = flask.Flask(__name__)
    app.config["DEBUG"] = True
    app.applications = applications

    @app.route("/", methods=["GET"])
    def home():
        return "<h1>The Home of Maestro Server</p>"

    @app.route("/get_batch_input", methods=["POST"])
    def get_batch_input():
        print("recieved!")
        print(request.form["uids"])

        return {"sample": 1, "sample2": 2}

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

    print("server running")
    app.run()


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
