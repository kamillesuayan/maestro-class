import os
import numpy as np
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    glue_compute_metrics,
)
import torch
import json

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.pipeline import (
    Scenario,
    DefenseAccess,
)

from Maestro.pipeline import AutoPipelineForVision, Scenario, AttackerAccess

# ------------------ LOCAL IMPORTS ---------------------------------


def compute_metrics_accuracy(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics("sst-2", preds, p.label_ids)


def load_all_applications(applications_config_path: str):
    print(f"Loading {applications_config_path} ......")
    application_list = {}
    # GeneticAttack
    print("Setting up the FGSM Attack pipeline....")
    with open(applications_config_path,"r") as f:
        application_configs = json.load(f)
    for app_config in application_configs["Application"]:
        print(app_config)
        name = app_config["name"]
        device = app_config["GPU"]
        device = torch.device(device if (torch.cuda.is_available()) else "cpu")
        myscenario = Scenario()
        attacker_access_yaml = app_config["attacker_access_yaml"]
        myscenario.load_from_yaml(attacker_access_yaml)
        dataset_name = app_config["dataset"]
        model_name = app_config["model"]["name"]
        checkpoint_path = app_config["model"]["checkpoint"]
        whether_finetune = True
        if checkpoint_path == "":
            whether_finetune = False
        print(model_name, checkpoint_path,device)
        pipeline = AutoPipelineForVision.initialize(
            name,
            dataset_name,
            model_name,
            checkpoint_path,
            myscenario,
            training_process=None,
            device=device,
            finetune=whether_finetune,
        )
        application_list[name] = pipeline

    # application_list["GeneticAttack"] = pipeline2
    # if "Adv_Training" in applications:
    #     print("Setting up the Adv_Training Attack pipeline....")
    #     name = "Adv_Training_example_model"
    #     dataset_name = "MNIST"
    #     myscenario = Scenario()
    #     myscenario.load_from_yaml("Attacker_Access/FGSM.yaml")
    #     # checkpoint_path = "models_temp/"
    #     # model_path = checkpoint_path + "lenet_mnist_model.pth"
    #     model_path = ''
    #     device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    #     pipeline2 = AutoPipelineForVision.initialize(
    #         name,
    #         dataset_name,
    #         model_path,
    #         '',
    #         compute_metrics_accuracy,
    #         myscenario,
    #         training_process=None,
    #         device=device,
    #         finetune=False,
    #     )
    #     application_list["Adv_Training"] = pipeline2

    return application_list
