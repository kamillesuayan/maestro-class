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

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.pipeline import (
    AutoPipelineAugmentedCV,
    AutoPipelineLossFuncCV,
    AugmentedPipelineCV,
    LossFuncPipelineCV,
    Scenario,
    DefenseAccess
)

from Maestro.pipeline import (
    AutoPipelineForVision,
    Scenario,
    AttackerAccess
)
# ------------------ LOCAL IMPORTS ---------------------------------


def compute_metrics_accuracy(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics("sst-2", preds, p.label_ids)


def load_all_applications(applications: List[str]):
    print(applications)
    application_list = {}
    # Data_Augmentation_CV
    if "Data_Augmentation_CV" in applications:
        print("Setting up the Data Augmentation CV pipeline....")
        name = "Data_Augmentation"
        dataset_name = "MNIST"
        myscenario = Scenario()
        myscenario.load_from_yaml("Defense_Access/CV_Data_Augmentation.yaml")
        checkpoint_path = "models_temp/"
        model_path = checkpoint_path + "lenet_mnist_model.pth"
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        pipeline2 = AutoPipelineAugmentedCV.initialize(
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
        application_list["Data_Augmentation_CV"] = pipeline2
    # Loss_Function_CV
    if "Loss_Function_CV" in applications:
        print("Setting up the Input Encoding CV pipeline....")
        name = "Loss_Function_CV"
        dataset_name = "MNIST"
        myscenario = Scenario()
        myscenario.load_from_yaml("Defense_Access/Loss_Function_CV.yaml")
        checkpoint_path = "models_temp/"
        model_path = checkpoint_path + "lenet_mnist_model.pth"
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        pipeline2 = AutoPipelineLossFuncCV.initialize(
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
        application_list["Loss_Function_CV"] = pipeline2
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
    if "Adv_Training" in applications:
        print("Setting up the Adv_Training Attack pipeline....")
        name = "Adv_Training_example_model"
        dataset_name = "MNIST"
        myscenario = Scenario()
        myscenario.load_from_yaml("Attacker_Access/FGSM.yaml")
        # checkpoint_path = "models_temp/"
        # model_path = checkpoint_path + "lenet_mnist_model.pth"
        model_path = ''
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
            finetune=False,
        )
        application_list["Adv_Training"] = pipeline2

    return application_list
