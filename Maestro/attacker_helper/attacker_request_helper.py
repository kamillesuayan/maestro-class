import numpy as np
import requests
import json
import pickle
import os
from typing import List, Iterator, Dict, Tuple, Any, Type
import random


class virtual_model:
    """
    model_wraper is an object that only contains method/data that are allowed to the users.
    """

    def __init__(self, request_url, application_name) -> None:
        self.request_url = request_url
        self.application_name = application_name
        self.batch_output_count = 0
        self.batch_gradient_count = 0

    def get_batch_output(self, perturbed_tokens, labels=[]):
        self.batch_output_count += 1
        return self._process_batch(self.request_url, perturbed_tokens, gradient=False,)

    def get_batch_input_gradient(self, perturbed_tokens, labels):
        self.batch_gradient_count += 1
        return self._process_batch(
            self.request_url, perturbed_tokens, labels, gradient=True,
        )

    def get_ref_image(self, target_label):
        data_file = (
            "./data_" + self.application_name + "_perturb_.pkl"
        )
        dev_data = []
        data = {
            "Application_Name": self.application_name,
            "data_type": "validation",
            "perturbation": "",
        }
        final_url = "{0}/get_data".format(self.request_url)
        response = requests.post(final_url, data=data)
        returned_json = response.json()
        for instance in returned_json["data"]:
            new_instance = {}
            for field in instance:
                new_instance[field] = instance[field]
            dev_data.append(new_instance)
        targeted_dev_data = []
        for instance in dev_data:
            if instance["label"] == target_label:
                targeted_dev_data.append(instance)
        return random.choice(targeted_dev_data)

    def _process_batch(self, url, batch, labels=[], gradient=False):
        """
        batch: batch to process, has the shape of [batch_size, channel, height of image, width of image]

        """
        # print(batch)
        batch = self.convert_np_matrix_to_list(batch)
        labels = self.convert_np_matrix_to_list(labels)
        payload = {
            "Application_Name": self.application_name,
            "data": batch,
            "labels": labels,
        }
        final_url = url + "/get_batch_output"
        if gradient:
            final_url = url + "/get_batch_input_gradient"
        response = requests.post(final_url, json=payload)
        outputs = json.loads(response.json()["outputs"])
        return np.array(outputs)

    # For NLP applications
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

    def convert_np_matrix_to_list(self, arr):
        arr = np.array(arr)
        return arr.tolist()

    # ------------------ AUGMENTED DEFENSE FUNCTIONS ---------------------------
    def send_augmented_dataset(self, train_set, defender):
        # print(defender)
        # defender = defender()
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

    # ------------------ AUGMENTED DEFENSE FUNCTIONS ---------------------------

    # ------------------ INPUT ENCODING DEFENSE FUNCTIONS ----------------------
    def send_detector_model(self, defender):
        model = defender.detector()
        payload = {
            "Application_Name": self.application_name,
            "model": model.state_dict(),
        }
        final_url = self.request_url + "/send_detector_model"
        response = requests.post(final_url, json=payload)
        returned_json = response.json()
        return returned_json

    # ------------------ INPUT ENCODING DEFENSE FUNCTIONS ----------------------

    # function that make a model switch its weights
    def switch_weights(self, file_path):
        data = {
            "Application_Name": self.application_name,
            "file_path": file_path,
        }
        final_url = "{0}/switch_weights".format(self.request_url)
        response = requests.post(final_url, data=data)
        return response
