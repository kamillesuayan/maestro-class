import numpy as np
import requests
import json
import pickle
import os
from typing import List, Iterator, Dict, Tuple, Any, Type


class virtual_model:
    """
    model_wraper is an object that only contains method/data that are allowed to the users.
    """

    def __init__(self, request_url, application_name) -> None:
        self.request_url = request_url
        self.application_name = application_name

    def get_batch_output(self, perturbed_tokens, labels):
        return self._process_batch(
            self.request_url, perturbed_tokens, labels, gradient=False,
        )

    def get_batch_input_gradient(self, perturbed_tokens, labels):
        return self._process_batch(
            self.request_url, perturbed_tokens, labels, gradient=True,
        )

    def get_embedding(self):
        embedding_file = "./embedding.pkl"
        if os.path.isfile(embedding_file):
            embedding = pickle.load(open(embedding_file, "rb"))
        else:

            data = {"Application_Name": self.application_name}
            final_url = "{0}/get_model_embedding".format(self.request_url)
            response = requests.post(final_url, data=data)
            retruned_json = json.loads(response.json()["data"])
            embedding = retruned_json
            print("embedding", len(embedding))
            with open(embedding_file, mode="wb") as f:
                pickle.dump(
                    embedding, f,
                )
        return embedding

    def convert_tokens_to_ids(self, text):
        data = {"Application_Name": self.application_name, "text": text}
        final_url = "{0}/convert_tokens_to_ids".format(self.request_url)
        response = requests.post(final_url, data=data)
        retruned_json = response.json()
        return retruned_json["data"]

    def convert_ids_to_tokens(self, id):
        data = {"Application_Name": self.application_name, "text": id}
        final_url = "{0}/convert_ids_to_tokens".format(self.request_url)
        response = requests.post(final_url, data=data)
        retruned_json = response.json()
        return retruned_json["data"]

    def get_data(self, data_type="validation"):
        data_file = "./data_" + self.application_name + ".pkl"
        dev_data = []
        print("getting data", data_file, os.path.isfile(data_file))
        if os.path.isfile(data_file):
            print("found local data, loading...")
            dev_data = pickle.load(open(data_file, "rb"))
        else:
            data = {
                "Application_Name": self.application_name,
                "data_type": data_type,
            }
            final_url = "{0}/get_data".format(self.request_url)
            response = requests.post(final_url, data=data)
            retruned_json = response.json()
            for instance in retruned_json["data"]:
                new_instance = {}
                for field in instance:
                    if isinstance(instance[field], List):
                        new_instance[field] = instance[field]
                    else:
                        new_instance[field] = instance[field]
                dev_data.append(new_instance)
            with open(data_file, mode="wb") as f:
                pickle.dump(
                    dev_data, f,
                )
        return dev_data

    def _process_batch(self, url, batch, labels, gradient=False):
        # if labels == None:
        #     labels = np.array([])
        payload = {
            "Application_Name": self.application_name,
            "data": batch.tolist(),
            "labels": labels.tolist(),
        }
        final_url = url + "/get_batch_output"
        if gradient:
            final_url = url + "/get_batch_input_gradient"
        response = requests.post(final_url, json=payload)
        print(response)
        outputs = json.loads(response.json()["outputs"])
        return outputs

    # ------------------ DEFENSE FUNCTIONS ------------------------------
    def send_augmented_dataset(self, train_set, defender):
        augmented_dataset = defender.defense(train_set)
        payload = {
            "Application_Name": self.application_name,
            "data": augmented_dataset,
        }
        final_url = self.request_url + "/send_augmented_dataset"
        response = requests.post(final_url, json=payload)
        returned_json = response.json()
        return returned_json

    def send_train_signal(self):
        # if labels == None:
        #     labels = np.array([])
        payload = {
            "Application_Name": self.application_name,
            "train": True,
        }
        final_url = self.request_url + "/send_train_signal"
        response = requests.post(final_url, json=payload)
        returned_json = response.json()
        return returned_json

    # ------------------ DEFENSE FUNCTIONS ------------------------------
