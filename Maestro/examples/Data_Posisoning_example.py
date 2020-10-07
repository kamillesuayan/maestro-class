from pipeline import Pipeline,Scenario, Attacker,model_wrapper
from Universal_Triggers_example import get_accuracy
from model import LstmClassifier

from typing import List, Iterator, Dict, Tuple, Any, Type
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
import torch

def test(model_wrapper, device, num_tokens_change,vocab):
    data_poision = None # predefined, since this is only testing applying the model.
    test_data = model_wrapper.get_test_data()
    model_wrapper.change_train_data("front",5,data_poision)
    model_wrapper.train()

    get_accuracy(model_wrapper,test_data,vocab,None,True,False)

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
    training_process = None
    device = 1
     # initialize Atacker, which specifies access rights
    training_data_access = 2
    dev_data_access = 0
    test_data_access = 1
    model_access = 0
    output_access = 0
    myattacker = Attacker(training_data_access,dev_data_access,test_data_access,model_access,output_access)

    # initialize Scenario. This defines our target
    target = None
    myscenario = Scenario(target,myattacker)

    model_wrapper = Pipeline(myscenario,train_data,dev_data,test_data,model,training_process,device).get_object()
    test(model_wrapper, device, 5,vocab)

if __name__ == "__main__":
    main()