import torch
import torch.nn as nn
from typing import List, Iterator, Dict, Tuple, Any, Type, Union
import torch.nn as nn
import torch.nn.functional as F
import transformers
import textattack
from textattack.models.helpers import GloveEmbeddingLayer
from textattack.models.helpers.utils import load_cached_state_dict
from textattack.shared import utils

# from pytorch_lightning.core.lightning import LightningModule


def build_model(
    model_name,
    num_labels: int,
    max_length: int = 128,
    device: int = 0,
    pretrained_file: str = None,
):
    if model_name == "FGSM_example_model":
        model = FGSM_example_model()
        if pretrained_file != None:
            model.load_state_dict(torch.load(pretrained_file, map_location="cpu"))
        return model
    elif model_name == "LSTM":
        model = LSTMForClassification(
            max_seq_length=max_length, num_labels=num_labels, emb_layer_trainable=False,
        )
        if pretrained_file:
            model.load_from_disk(pretrained_file)
        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)
        return model
    else:
        config = transformers.AutoConfig.from_pretrained(
            model_name, num_labels=num_labels, finetuning_task="imdb"
        )
        # config.architectures = ["BertForSequenceClassification"]
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        ).to(device)
        # model = transformers.AutoModelForSequenceClassification.from_config(
        #     config=config
        # ).to(device)

        # tokenizer = textattack.models.tokenizers.AutoTokenizer(
        #     model_name, use_fast=True, max_length=max_length
        # )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        print(tokenizer)
        model = Model_and_Tokenizer(model, tokenizer)
        return model


class Model_and_Tokenizer(nn.Module):
    def __init__(self, model, tokenizer) -> None:
        super(Model_and_Tokenizer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer


class BasicClassifier(nn.Module):
    def __init__(self, model_name, pretrained_file: str, regularizer) -> None:
        super(BasicClassifier, self).__init__()
        self._regularizer = regularizer
        self.model_name = model_name
        self.pretrained_file = pretrained_file

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:
        """
        Computes the regularization penalty for the model.
        Returns 0 if the model was not configured to use regularization.
        """
        if self._regularizer is None:
            return 0.0
        else:
            return self._regularizer(self)


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


# LightningModule
class LSTMForClassification(BasicClassifier):
    """A long short-term memory neural network for text classification.
    We use different versions of this network to pretrain models for
    text classification.
    """

    def __init__(
        self,
        hidden_size=150,
        depth=1,
        dropout=0.3,
        num_labels=2,
        max_seq_length=128,
        model_path=None,
        emb_layer_trainable=True,
    ):
        super(LSTMForClassification, self).__init__("LSTMForClassification", None, None)
        if depth <= 1:
            # Fix error where we ask for non-zero dropout with only 1 layer.
            # nn.module.RNN won't add dropout for the last recurrent layer,
            # so if that's all we have, this will display a warning.
            dropout = 0
        self.drop = nn.Dropout(dropout)
        self.emb_layer_trainable = emb_layer_trainable
        self.emb_layer = GloveEmbeddingLayer(emb_layer_trainable=emb_layer_trainable)
        self.word2id = self.emb_layer.word2id
        self.encoder = nn.LSTM(
            input_size=self.emb_layer.n_d,
            hidden_size=hidden_size // 2,
            num_layers=depth,
            dropout=dropout,
            bidirectional=True,
        )
        d_out = hidden_size
        self.out = nn.Linear(d_out, num_labels)
        self.tokenizer = textattack.models.tokenizers.GloveTokenizer(
            word_id_map=self.word2id,
            unk_token_id=self.emb_layer.oovid,
            pad_token_id=self.emb_layer.padid,
            max_length=max_seq_length,
        )

        if model_path is not None:
            self.load_from_disk(model_path)

    def load_from_disk(self, model_path):
        self.load_state_dict(load_cached_state_dict(model_path))
        self.word_embeddings = self.emb_layer.embedding
        self.lookup_table = self.emb_layer.embedding.weight.data
        self.to(utils.device)
        self.eval()

    def forward(self, sentence, labels):
        # ensure RNN module weights are part of single contiguous chunk of memory
        self.encoder.flatten_parameters()

        emb = self.emb_layer(sentence.t())
        emb = self.drop(emb)

        output, hidden = self.encoder(emb)
        output = torch.max(output, dim=0)[0]

        output = self.drop(output)
        pred = self.out(output)
        return pred

    def training_step(self, batch, batch_idx):
        print("in training_step", batch)
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss
