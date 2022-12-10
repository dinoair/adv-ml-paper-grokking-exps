import torch
from torch.utils.data import Dataset


class Text2SparqlDataset(Dataset):
    def __init__(self, tokenized_question_list, tokenized_sparql_list,
                 question_list, sparql_list, dev):
        self.tokenized_question_list = tokenized_question_list
        self.tokenized_sparql_list = tokenized_sparql_list
        self.question_list = question_list
        self.sparql_list = sparql_list
        self.device = dev

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx):
        nl_tokens, token_type_ids, nl_attention_mask, original_question = self.tokenized_question_list['input_ids'][
                                                                              idx].to(self.device), \
                                                                          self.tokenized_question_list[
                                                                              'token_type_ids'][idx].to(self.device), \
                                                                          self.tokenized_question_list[
                                                                              'attention_mask'][idx].to(self.device), \
                                                                          self.question_list[idx]

        sparql_tokens, sparql_query = self.tokenized_sparql_list[idx], self.sparql_list[idx]
        sparql_tokens = torch.tensor(sparql_tokens).to(self.device)
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
