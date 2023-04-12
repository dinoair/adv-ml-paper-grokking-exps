import torch
from torch.utils.data import Dataset


class Text2QueryDataset(Dataset):
    def __init__(self, tokenized_question_list, tokenized_query_list,
                 question_list, query_list, model_type, tokenizer, dev):
        self.tokenized_question_list = tokenized_question_list
        self.tokenized_query_list = tokenized_query_list
        self.question_list = question_list
        self.query_list = query_list
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
            query_tokens = torch.tensor(self.tokenized_query_list['input_ids'][idx]).to(self.device)
            query_tokens[query_tokens == self.tokenizer.pad_token_id] == -100
        elif self.model_type == 'vanilla':
            query_tokens = torch.tensor(self.tokenized_query_list[idx]).to(torch.long).to(self.device)

        query = self.query_list[idx]

        return {
            "nl": {
                "input_ids": nl_tokens,
                "token_type_ids": token_type_ids,
                "attention_mask": nl_attention_mask,
                "original_question": original_question
            },
            "query": {
                "input_ids": query_tokens,
                "original_query": query
            }
        }
