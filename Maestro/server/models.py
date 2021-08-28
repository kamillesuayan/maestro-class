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
    AutoPipelineForVision,
    VisionPipeline,
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
    if "FGSM" in applications:
        print("Setting up the FGSM Attack pipeline....")
        name = "FGSM_example_model"
        dataset_name = "MNIST"
        myscenario = Scenario()
        myscenario.load_from_yaml("Defense_Access/FGSM.yaml")
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

    return application_list
