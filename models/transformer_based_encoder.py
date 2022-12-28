import sys
from copy import deepcopy

import torch.nn as nn


class TransformerBasedEncoder(nn.Module):
    def __init__(self, bert_model, n_last_layers2train):
        super(TransformerBasedEncoder, self).__init__()
        self.bert_module = deepcopy(bert_model)
        self.disable_bert_training()

        if 1 <= n_last_layers2train < len(self.bert_module.encoder.layer):
            self.modules2train = [*self.bert_module.encoder.layer[-n_last_layers2train:], self.bert_module.pooler]
        elif n_last_layers2train == len(self.bert_module.encoder.layer):
            self.modules2train = [*self.bert_module.encoder.layer[-n_last_layers2train:],
                                  self.bert_module.pooler, self.bert_module.embeddings]
        elif n_last_layers2train == 0:
            self.modules2train = [self.bert_module.pooler]
        elif n_last_layers2train == -1:
            self.modules2train = []
        else:
            print('Wrong params amount!')
            sys.exit()

    def forward(self, input_batch):
        bert_output = self.bert_module(
            **{key: input_batch[key] for key in ['input_ids', 'token_type_ids', 'attention_mask']})

        return {
            "last_hiddens": bert_output.last_hidden_state,
            "pooler": bert_output.pooler_output
        }

    def disable_bert_training(self):
        for module in self.bert_module.parameters():
            module.requires_grad = False

    def enable_bert_layers_training(self):
        for module in self.modules2train:
            for param in module.parameters():
                param.requires_grad = True

        # for name, param in self.bert_module.named_parameters():
        #     print(name, param.requires_grad)
