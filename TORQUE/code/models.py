from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel, BertModel

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_m\
odel.bin",
}


class MultitaskClassifier(BertPreTrainedModel):
    def __init__(self, config, mlp_hid=16):
        super(MultitaskClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        outputs = self.dropout(outputs[0])
        # QA MLP                                                                
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)
        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits


class MultitaskClassifierRoberta(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, mlp_hid=16):
        super(MultitaskClassifierRoberta, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None,
                output_hidden_states=False):
        output_tuple = self.roberta(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask)

        outputs = self.dropout(output_tuple[0])
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)
        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)

        results = [logits]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            results.append(loss)

        if output_hidden_states:
            results.append(vectors)

        return results


class PoeClassifier(nn.Module):
    def __init__(self, model_type, robust_model_name_or_path, bias_model_name_or_path,
                 bias_model_state_dict,
                 poe_mode="poe", mlp_hid=16, cache_dir=None, entropy_weight=0.1):
        super().__init__()
        self.poe_mode = poe_mode
        assert self.poe_mode in ["poe", "poe_mixin", "poe_mixin_h"]
        if model_type == "roberta":
            classifier_class = MultitaskClassifierRoberta
        elif model_type == "bert":
            classifier_class = MultitaskClassifier
        else:
            raise ValueError("model_type must be either roberta or bert")
        self.bias_model = classifier_class.from_pretrained(bias_model_name_or_path, cache_dir=cache_dir,
                                                           mlp_hid=mlp_hid, state_dict=bias_model_state_dict)
        self.robust_model = classifier_class.from_pretrained(robust_model_name_or_path, cache_dir=cache_dir,
                                                             mlp_hid=mlp_hid)
        if self.poe_mode in {"poe_mixin", "poe_mixin_h"}:
            self.bias_weight_linear = nn.Linear(self.robust_model.config.hidden_size, 1)
        else:
            self.bias_weight_linear = None
        self.bias_model.eval()
        self.robust_model.train()
        self.entropy_weight = entropy_weight
        self.num_labels = 2

    def forward(self, input_ids, offsets, lengths, attention_mask,
                token_type_ids, labels):
        with torch.no_grad():
            bias_logits, bias_hidden_states = self.bias_model(input_ids, offsets, lengths,
                                                              attention_mask=attention_mask,
                                                              token_type_ids=token_type_ids,
                                                              output_hidden_states=True)
        robust_logits, robust_hidden_states = self.robust_model(input_ids, offsets, lengths,
                                                                attention_mask=attention_mask,
                                                                token_type_ids=token_type_ids,
                                                                output_hidden_states=True)
        softmax = nn.Softmax(dim=-1)
        softplus = nn.Softplus()
        bias_prob = softmax(bias_logits)
        robust_prob = softmax(robust_logits)

        entropy_loss = None
        if self.poe_mode == "poe":
            logits = torch.log2(robust_prob) + torch.log2(bias_prob)
        elif self.poe_mode in {"poe_mixin", "poe_mixin_h"}:
            # bias_hidden_states = torch.sum(bias_hidden_states * attention_mask.unsqueeze(2),
            #                                dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
            bias_weight = self.bias_weight_linear(bias_hidden_states)
            bias_weight = softplus(bias_weight)
            # unpack bias_weight

            logits = torch.log2(robust_prob) + bias_weight * torch.log2(bias_prob)
            if self.poe_mode == "poe_mixin_h":
                prob = softmax(bias_weight * torch.log2(bias_prob))
                entropy_loss = torch.sum(-prob * torch.log2(prob), dim=-1)
                entropy_loss = torch.mean(entropy_loss)
        else:
            raise ValueError("Wrong peo mode is passed")

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if entropy_loss is not None:
            loss += self.entropy_weight * entropy_loss

        return logits, loss

    def train(self):
        super().train()
        self.bias_model.eval()


class AFLiteRoberta(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(AFLiteRoberta, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None,
                output_hidden_states=False):
        output_tuple = self.roberta(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask)

        outputs = output_tuple[0]
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)

        return vectors

