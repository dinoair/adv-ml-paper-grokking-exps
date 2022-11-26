import torch
from copy import deepcopy
import torch.nn as nn


class TransformerBasedEncoder(nn.Module):
    def __init__(self, bert_model):
        super(TransformerBasedEncoder, self).__init__()
        self.bert_module = deepcopy(bert_model)
        
        

    def forward(self, input_batch):
        bert_output = self.bert_module(
            **{key: input_batch[key] for key in ['input_ids', 'token_type_ids', 'attention_mask']})

        return {
            "last_hiddens": bert_output.last_hidden_state,
            "pooler": bert_output.pooler_output
        }

    def disable_bert_training(self):
        for layer in self.bert_module.encoder.layer:
            for param in layer.parameters():
                param.requires_grad = False
        self.bert_module.pooler.requires_grad_(False)

    def enable_some_bert_layers_training(self, layers2freeze):
        layers4freeze = [*[self.bert_module.encoder.layer[:layers2freeze]] + [self.bert_module.embeddings]]
        layers2train = [*self.bert_module.encoder.layer[layers2freeze - len(self.bert_module.encoder.layer):]]
        for layer in layers4freeze:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in layers2train:
            for param in layer.parameters():
                param.requires_grad = True
                
        self.bert_module.pooler.requires_grad_(True)