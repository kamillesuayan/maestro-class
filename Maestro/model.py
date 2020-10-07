import torch
import torch.nn as nn
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from typing import List, Iterator, Dict, Tuple, Any, Type
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file


def build_model(model_name, pretrained_file: str, vocab=None):
    if model_name == "FGSM_example_model":
        model = FGSM_example_model()
        if pretrained_file != None:
            model.load_state_dict(torch.load(pretrained_file, map_location="cpu"))
        return model
    elif model_name == "LSTM":
        embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        weight = _read_pretrained_embeddings_file(
            embedding_path, embedding_dim=300, vocab=vocab, namespace="tokens"
        )
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size("tokens"),
            embedding_dim=300,
            weight=weight,
            trainable=False,
        )
        word_embedding_dim = 300
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        encoder = PytorchSeq2VecWrapper(
            torch.nn.LSTM(
                word_embedding_dim, hidden_size=512, num_layers=2, batch_first=True
            )
        )
        model = LstmClassifier(word_embeddings, encoder, vocab)
        if pretrained_file != None:
            model.load_state_dict(torch.load(pretrained_file, map_location="cpu"))
        return model


class model(nn.Module):
    def __init__(self, vocab) -> None:
        self.vocab = vocab


class BasicClassifier(model):
    def __init__(self, model_name, pretrained_file: str) -> None:
        super().__init__()


class FGSM_example_model(nn.Module):
    def __init__(self) -> None:
        super(FGSM_example_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LstmClassifier(Model):
    def __init__(self, word_embeddings: nn.Module, encoder: nn.Module, vocab) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size("labels"),
        )
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label) -> Dict[str, Any]:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False) -> Dict[str, Any]:
        return {"accuracy": self.accuracy.get_metric(reset)}

