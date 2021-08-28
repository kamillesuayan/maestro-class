import os
import numpy as np
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
from Maestro.pipeline import (
    AutoPipelineForNLP,
    AutoPipelineForVision,
    Pipeline,
    Scenario,
    AttackerAccess,
    AutoPipelineForSec,
)
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    glue_compute_metrics,
)
import torch


def compute_metrics_accuracy(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics("sst-2", preds, p.label_ids)


def load_all_applications(applications: List[str]):
    print(applications)
    application_list = {}

    # Universal Triggers & Hotflip
    if ("Universal_Attack" in applications) | ("Hotflip" in applications):
        bert = True
        checkpoint_path = ""
        dataset_name = "SST2"
        if bert:
            checkpoint_path = "models_temp/" + "BERT_sst2_label/"
            name = "bert-base-uncased"
        else:
            checkpoint_path = "models_temp/" + "textattackLSTM/"
            name = "LSTM"
        model_path = checkpoint_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        myscenario = Scenario()
        myscenario.load_from_yaml("Attacker_Access/Universal_Trigger.yaml")
        print(myscenario)
        print("Settting up the Universal Triggers Attack pipeline....")
        pipeline = AutoPipelineForNLP.initialize(
            name,
            dataset_name,
            model_path,
            checkpoint_path,
            compute_metrics_accuracy,
            myscenario,
            training_process=None,
            device=0,
            finetune=True,
        )
        application_list["Universal_Attack"] = pipeline
        application_list["Hotflip"] = pipeline
        application_list["Data_Poisoning"] = pipeline

    # FGSM
    if "FGSM" in applications:
        print("Setting up the FGSM Attack pipeline....")
        name = "FGSM_example_model"
        dataset_name = "MNIST"
        myscenario = Scenario()
        myscenario.load_from_yaml("Attacker_Access/FGSM.yaml")
        checkpoint_path = "models_temp/"
        model_path = checkpoint_path + "lenet_mnist_model.pth"
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        pipeline2 = AutoPipelineForVision.initialize(
            name,
            dataset_name,
            model_path,
            checkpoint_path,
            compute_metrics_accuracy,
            myscenario,
            training_process=None,
            device=device,
            finetune=True,
        )
        application_list["FGSM"] = pipeline2

    # Security attack Malimg
    if "Malimg" in applications:
        print("Setting up the Malimg Attack pipeline....")
        name = "MalimgClassifier"
        dataset_name = "Malimg"
        myscenario = Scenario()
        myscenario.load_from_yaml("Attacker_Access/FGSM.yaml")
        checkpoint_path = "models_temp/malimg/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        model_path = checkpoint_path + "malimg.pth"

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        # pipeline2 = AutoPipelineForVision.initialize(
        #     name,
        #     dataset_name,
        #     model_path,
        #     checkpoint_path,
        #     compute_metrics_accuracy,
        #     myscenario,
        #     training_process=None,
        #     device=device,
        #     finetune=True,
        # )

        pipeline = AutoPipelineForSec.initialize(
            name,
            dataset_name,
            model_path,
            checkpoint_path,
            compute_metrics_accuracy,
            myscenario,
            training_process=True,
            device=0,
            finetune=False,
        )

        application_list["Malimg"] = pipeline


    return application_list
