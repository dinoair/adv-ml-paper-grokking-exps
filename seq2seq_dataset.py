import torch
from torch.utils.data import Dataset


class Text2SparqlDataset(Dataset):
    def __init__(self, tokenized_question_list, tokenized_sparql_list,
                 question_list, sparql_list, model_type, tokenizer, dev):
        self.tokenized_question_list = tokenized_question_list
        self.tokenized_sparql_list = tokenized_sparql_list
        self.question_list = question_list
        self.sparql_list = sparql_list
        self.device = dev
        self.model_type = model_type
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx):
        nl_tokens, token_type_ids, nl_attention_mask, original_question = torch.tensor(self.tokenized_question_list['input_ids'][
                                                                              idx]).to(self.device), \
                                                                          torch.tensor(self.tokenized_question_list[
                                                                              'token_type_ids'][idx]).to(self.device), \
                                                                          torch.tensor(self.tokenized_question_list[
                                                                              'attention_mask'][idx]).to(self.device), \
                                                                          self.question_list[idx]

        if self.model_type == 't5':
            sparql_tokens = torch.tensor(self.tokenized_sparql_list['input_ids'][idx]).to(self.device)
            sparql_tokens[sparql_tokens == self.tokenizer.pad_token_id] == -100
        elif self.model_type == 'vanilla':
            sparql_tokens = torch.tensor(self.tokenized_sparql_list[idx]).to(torch.long).to(self.device)

        sparql_query = self.sparql_list[idx]

        return {
            "nl": {
                "input_ids": nl_tokens,
                "token_type_ids": token_type_ids,
                "attention_mask": nl_attention_mask,
                "original_question": original_question
            },
            "sparql": {
                "input_ids": sparql_tokens,
                "original_query": sparql_query
            }
        }
