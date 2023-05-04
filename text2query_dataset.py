import torch
from torch.utils.data import Dataset


class Text2QueryDataset(Dataset):
    def __init__(self, tokenized_input_list, tokenized_target_list,
                 question_list, query_list, model_type, tokenizer, dev, environment_list=None):
        self.tokenized_input_list = tokenized_input_list
        self.tokenized_target_list = tokenized_target_list
        self.question_list = question_list
        self.query_list = query_list
        self.device = dev
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.environment_list = environment_list

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx):
        nl_tokens, token_type_ids, nl_attention_mask, original_question = torch.tensor(self.tokenized_input_list['input_ids'][
                                                                              idx]).to(self.device), \
                                                                          torch.tensor(self.tokenized_input_list[
                                                                              'token_type_ids'][idx]).to(self.device), \
                                                                          torch.tensor(self.tokenized_input_list[
                                                                              'attention_mask'][idx]).to(self.device), \
                                                                          self.question_list[idx]

        if self.model_type == 't5':
            query_tokens = torch.tensor(self.tokenized_target_list['input_ids'][idx]).to(self.device)
            query_tokens[query_tokens == self.tokenizer.pad_token_id] == -100
        elif self.model_type == 'vanilla':
            query_tokens = torch.tensor(self.tokenized_target_list[idx]).to(torch.long).to(self.device)

        query = self.query_list[idx]

        dataset_dict = {
            "input": {
                "input_ids": nl_tokens,
                "token_type_ids": token_type_ids,
                "attention_mask": nl_attention_mask,
                "original_question": original_question
            },
            "target": {
                "input_ids": query_tokens,
                "original_query": query
            }
        }

        if self.environment_list is not None:
            dataset_dict['target']['env_name'] = self.environment_list[idx]

        return dataset_dict
