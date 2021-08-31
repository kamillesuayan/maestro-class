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
    AutoPipelineInputEncodingCV,
    AugmentedPipelineCV,
    InputEncodingPipelineCV,
    Scenario,
    DefenseAccess
)
# ------------------ LOCAL IMPORTS ---------------------------------


def compute_metrics_accuracy(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics("sst-2", preds, p.label_ids)


def load_all_applications(applications: List[str]):
    print(applications)
    application_list = {}
    # FGSM
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
    if "Input_Encoding_CV" in applications:
        print("Setting up the Input Encoding CV pipeline....")
        name = "Input_Encoding_CV"
        dataset_name = "MNIST"
        myscenario = Scenario()
        myscenario.load_from_yaml("Defense_Access/CV_Input_Encoding.yaml")
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
        application_list["Input_Encoding_CV"] = pipeline2

    return application_list
