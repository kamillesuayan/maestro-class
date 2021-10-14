from typing import List, Iterator, Dict, Tuple, Any, Type
from Maestro.data import HuggingFaceDataset, get_dataset
import torch
import transformers
def main():
    device=0
    model_name = "bert-base-uncased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model = model.from_pretrained("models_temp/" + "BERT_sst2_label/")
    test_data = HuggingFaceDataset(
        name="glue", subset="sst2", split="train", label_map=None, shuffle=True
    )
    print(len(test_data))
    # test_data.indexed(tokenizer,128)
    # test_data = test_data.get_json_data()
    polarity = 0
    print(test_data.label_names)
    negatives = []
    for i in range(len(test_data)):
        if test_data.examples[i][1] == polarity:
            negatives.append(test_data.examples[i])
        
    # print(negatives)
    D_clean = None
    D_adv = None
    D_eval = None
    D_poison = None
    


if __name__ == "__main__":
    main()