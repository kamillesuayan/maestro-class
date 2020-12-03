import os
import numpy as np
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
from Maestro.pipeline import (
    AutoPipelineForNLP,
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


def compute_metrics_accuracy(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics("sst-2", preds, p.label_ids)


def load_all_applications(applications: List[str]):
    print(applications)
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
    target = "Universal Perturbation"
    myscenario = Scenario(target, myattacker)
    pipeline = AutoPipelineForNLP.initialize(
        name,
        dataset_name,
        model_path,
        checkpoint_path,
        compute_metrics_accuracy,
        myscenario,
        training_process=None,
        device=1,
        finetune=True,
    )
    return {"Universal_Attack": pipeline.get_object()}
