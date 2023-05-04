import torch
import copy
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5Model
from transformers.optimization import Adafactor



class IRMT5Model(nn.Module):
    def __init__(self, model_config, device, tokenizer, compound_types_list):
        super(IRMT5Model, self).__init__()
        self.model_name = "t5"
        self.model_config = model_config
        self.device = device
        self.tokenizer = tokenizer

        hugginface_pretrained_model = model_config['model']
        self.t5_model = T5Model.from_pretrained(hugginface_pretrained_model).to(self.device)
        self.t5_model.resize_token_embeddings(len(tokenizer))
        # as in https://arxiv.org/pdf/1910.10683.pdf for fine-tuning
        self.t5_optimizer = Adafactor(self.t5_model.parameters(), lr=self.model_config['learning_rate'], relative_step=False)
        self.criterion = CrossEntropyLoss(ignore_index=-100)

        self.environment_head = dict()
        lm_head = nn.Linear(self.t5_model.config.d_model, len(self.tokenizer.index2word))
        for env_name in compound_types_list:
            self.environment_head[env_name] = copy.deepcopy(lm_head).to(self.device)

        self.env_optimizers_dict = dict()
        for env_name in compound_types_list:
            self.env_optimizers_dict[env_name] = Adafactor(self.environment_head[env_name].parameters(), lr=self.model_config['learning_rate'], relative_step=False)


    def train_on_batch(self, input_data, target_data, env_name):
        self.t5_model.train()

        self.t5_optimizer.zero_grad()
        self.env_optimizers_dict[env_name].zero_grad()

        input_ids, attention_mask = input_data['input_ids'], input_data['attention_mask']
        target_ids = target_data['input_ids']

        # batch_size, seq_len, dim
        t5_logits = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=target_ids).last_hidden_state
        env_head = self.environment_head[env_name]
        env_vocab_projection = env_head(t5_logits)
        loss = self.criterion(env_vocab_projection.view(-1, env_vocab_projection.size(-1)), target_ids.view(-1))

        loss.backward()
        self.t5_optimizer.step()
        self.env_optimizers_dict[env_name].step()

        return loss.item()

    def evaluate_batch(self, input_data, target_data, env_name):
        self.t5_model.eval()
        result_dict = dict()
        input_ids, attention_mask = input_data['input_ids'], input_data['attention_mask']
        target_ids = target_data['input_ids']
        with torch.no_grad():
            t5_logits = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=target_ids).last_hidden_state
            env_head = self.environment_head[env_name]
            env_vocab_projection = env_head(t5_logits)
            loss = self.criterion(env_vocab_projection.view(-1, env_vocab_projection.size(-1)), target_ids.view(-1))

        result_dict['loss'] = loss.item()

        if env_name == 'full':
            decoder_result_list = torch.argmax(env_vocab_projection, dim=-1).cpu().numpy()
            decoded_query_list = []
            for sample in decoder_result_list:
                decoded_query_tokens = self.tokenizer.decode(sample)
                query = " ".join(decoded_query_tokens)
                decoded_query_list.append(query)

            result_dict['predicted_query'] = decoded_query_list
        return result_dict
