from pipeline import Pipeline,Scenario, Attacker,model_wrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from copy import deepcopy
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy
import heapq
from operator import itemgetter

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.training.trainer import Trainer,GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import DataLoader
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import move_to_device
from allennlp.common.util import lazy_groups_of
# from torchvision import datasets, transforms


def get_accuracy(model_wrapper: model_wrapper, dev_data,vocab,trigger_token_ids, batch=True,triggers=False) -> None:       
    model_wrapper.model.get_metrics(reset=True)
    model_wrapper.model.eval() # model should be in eval() already, but just in case
    if batch:
        with torch.no_grad(): 
            batch = move_to_device(dev_data, cuda_device=1)
            model_wrapper.get_batch_output(batch)
    else:
        train_sampler = BucketBatchSampler(dev_data,batch_size=128, sorting_keys = ["tokens"])
        train_dataloader = DataLoader(dev_data, batch_sampler=train_sampler)
        model_wrapper.model.to(1)
        if triggers:
            print_string = ""
            for idx in trigger_token_ids:
                print_string = print_string + vocab.get_token_from_index(idx) + ', '
            with torch.no_grad(): 
                for batch in train_dataloader:
                    eval_with_triggers(model_wrapper, batch, trigger_token_ids,False)
        else:
            with torch.no_grad(): 
                for batch in train_dataloader:
                    batch = move_to_device(batch, cuda_device=1)
                    model_wrapper.get_batch_output(batch)
        

    print(model_wrapper.model.get_metrics(True)['accuracy'])
    model_wrapper.model.train()

def eval_with_triggers(model_wrapper: nn.Module, batch, trigger_token_ids: List[int], gradient=True) -> Dict[str, Any]:
    # if gradient is true, this function returns the gradient of the input with the appended trigger tokens
    trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
    with torch.cuda.device(1):
        trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch['label']), 1).cuda()
        original_tokens = batch['tokens']['tokens']["tokens"].clone().cuda()
    batch['tokens']['tokens']["tokens"] = torch.cat((trigger_sequence_tensor, original_tokens), 1)
    if gradient:
        data_grad = model_wrapper.get_batch_input_gradient(batch)
        batch['tokens']['tokens']["tokens"] = original_tokens
        return data_grad
    else:
        outputs = model_wrapper.get_batch_output(batch)
        batch['tokens']['tokens']["tokens"] = original_tokens
        return outputs



def test(model_wrapper, device,validation_sampler, num_tokens_change,vocab):
    test_data = model_wrapper.get_test_data()
    trigger_token_ids = [672, 290, 290]
    get_accuracy(model_wrapper,test_data,vocab,trigger_token_ids,False,True)

    
def main():
    use_cuda=True
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer},
                                                    use_subtrees=True)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    test_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt')
    vocab = Vocabulary.from_instances(train_data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    test_data.index_with(vocab)
    train_sampler = BucketBatchSampler(train_data,batch_size=32, sorting_keys = ["tokens"])
    validation_sampler = BucketBatchSampler(dev_data,batch_size=32, sorting_keys = ["tokens"])
    train_dataloader = DataLoader(train_data,batch_sampler=train_sampler)
    validation_dataloader = DataLoader(dev_data,batch_sampler=validation_sampler)

    embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
    weight = _read_pretrained_embeddings_file(embedding_path,
                                                embedding_dim=300,
                                                vocab=vocab,
                                                namespace="tokens")
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=300,
                                weight=weight,
                                trainable=False)
    word_embedding_dim = 300
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    # word_embeddings.requires_grad = True
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                  hidden_size=512,
                                                  num_layers=2,
                                                  batch_first=True))
    model = LstmClassifier(word_embeddings, encoder, vocab)
    model.to(1)

    model_path = "models/" + "LSTM/" + "model.th"
    vocab_path = "models/" + "LSTM/" + "vocab"
    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path):
        vocab = Vocabulary.from_files(vocab_path)
        model = LstmClassifier(word_embeddings, encoder, vocab)
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
    # otherwise train model from scratch and save its weights
    else:
        optimizer = optim.Adam(model.parameters())
        trainer = GradientDescentTrainer(model=model,
                        optimizer=optimizer,
                        data_loader=train_dataloader,
                        validation_data_loader = validation_dataloader,
                        num_epochs=8,
                        patience=1,
                        cuda_device=1)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)
    # model.train() #.cuda(1) # rnn cannot do backwards in train mode
   
    print("CUDA Available: ",torch.cuda.is_available())
    # print("accuracy: ",get_accuracy(model_wrapper,dev_data,vocab))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    training_process = None

    # initialize Atacker, which specifies access rights
    training_data_access = 0
    dev_data_access = 0
    test_data_access = 2
    model_access = 0
    output_access = 0
    myattacker = Attacker(training_data_access,dev_data_access,test_data_access,model_access,output_access)

    # initialize Scenario. This defines our target
    target = None
    myscenario = Scenario(target,myattacker)

    model_wrapper = Pipeline(myscenario,train_data,dev_data,test_data,model,training_process,device).get_object()
    test(model_wrapper, device,validation_sampler, 5,vocab)

if __name__ == "__main__":
    main()