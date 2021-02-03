import os
import numpy as np
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
from Maestro.pipeline import (
    AutoPipelineForNLP,
    AutoPipelineForVision,
    Pipeline,
    Scenario,
    Attacker,
    model_wrapper,
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
    # only doing this for Universal Attack for the moment
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
    training_process = None

    # initialize Atacker, which specifies access rights

    training_data_access = 0
    validation_data_access = 3
    test_data_access = 0
    model_access = 0
    output_access = 2
    myattacker = Attacker(
        training_data_access,
        validation_data_access,
        test_data_access,
        model_access,
        output_access,
    )

    # initialize Scenario. This defines our target
    # target = "Universal Perturbation"
    # myscenario = Scenario(target, myattacker)
    # print("Settting up the Universal Triggers Attack pipeline....")
    # pipeline = AutoPipelineForNLP.initialize(
    #     name,
    #     dataset_name,
    #     model_path,
    #     checkpoint_path,
    #     compute_metrics_accuracy,
    #     myscenario,
    #     training_process=None,
    #     device=0,
    #     finetune=True,
    # )
    # application_list["Universal_Attack"] = pipeline.get_object()

    # FGSM
    print("Settting up the FGSM Attack pipeline....")
    name = "FGSM_example_model"
    dataset_name = "MNIST"
    attacker_config = "Attacker_Access/FGSM.yaml"
    myattacker = Attacker()
    myattacker.load_from_yaml(attacker_config)
    target = ""
    myscenario = Scenario(target, myattacker)
    checkpoint_path = "models_temp/"
    model_path = checkpoint_path + "lenet_mnist_model.pth"
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    pipeline = AutoPipelineForVision.initialize(
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
    application_list["FGSM"] = pipeline.get_object()

    return application_list
