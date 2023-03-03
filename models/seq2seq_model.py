import torch
import numpy as np
from torch import nn
import torch.optim as optim
from transformers import get_constant_schedule_with_warmup
from transformers import AutoModel

from models.recurrent_decoder import RecurrentDecoder
from models.seq2seq_attention import Seq2seqAttention
from models.transformer_based_encoder import TransformerBasedEncoder
from sparql_tokenizer import SPARQLTokenizer


class Seq2seqModel(nn.Module):
    def __init__(self, model_config: dict, device: str, target_tokenizer: SPARQLTokenizer):
        super(Seq2seqModel, self).__init__()
        self.model_config = model_config
        self.device = device
        self.target_tokenizer = target_tokenizer
        self.batch_size = self.model_config['batch_size']

        hugginface_pretrained_model = self.model_config['model']
        transformer_based_model = AutoModel.from_pretrained(hugginface_pretrained_model)
        trainable_layers_num = self.model_config['n_last_layers2train']
        self.encoder: TransformerBasedEncoder = TransformerBasedEncoder(transformer_based_model,
                                                                        trainable_layers_num).to(self.device)

        decoder_hidden_state = self.encoder.bert_module.pooler.dense.weight.shape[0]
        self.decoder: RecurrentDecoder = RecurrentDecoder(hidden_size=decoder_hidden_state,
                                                          vocab_size=len(target_tokenizer.word2index)).to(self.device)

        self.softmax = nn.LogSoftmax(dim=1).to(self.device)

        model_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.AdamW(model_params, lr=self.model_config['learning_rate'])

        if self.model_config['enable_attention']:
            self.attention_module = Seq2seqAttention().to(self.device)
            self.optimizer.add_param_group({"params": self.attention_module.parameters()})
            self.vocab_projection_layer = nn.Linear(2 * decoder_hidden_state, len(target_tokenizer.word2index)).to(
                self.device)
            self.optimizer.add_param_group({"params": self.vocab_projection_layer.parameters()})
        else:
            self.vocab_projection_layer = nn.Linear(decoder_hidden_state, len(target_tokenizer.word2index)).to(
                self.device)
            self.optimizer.add_param_group({"params": self.vocab_projection_layer.parameters()})

        self.optimizer_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                                     num_warmup_steps=1000)
        self.criterion = nn.NLLLoss()

    def train_on_batch(self, input_data, target_data):
        self.optimizer.zero_grad()

        encoder_output = self.encoder(input_data)

        encoder_states = encoder_output['last_hiddens']
        pooler = encoder_output['pooler']

        decoder_input = torch.tensor([[0] * self.batch_size],
                                     dtype=torch.long, device=self.device).view(1, self.batch_size, 1)
        decoder_hidden = pooler.view(1, self.batch_size, -1)

        target_tensor = target_data['input_ids'].view(self.batch_size, self.target_tokenizer.max_sent_len, 1)

        target_length = target_tensor.shape[1]
        loss = 0.0
        for idx in range(target_length):
            decoder_output, decoder_hidden = self.decoder(input_data=decoder_input,
                                                          hidden_state=decoder_hidden,
                                                          batch_size=self.batch_size)
            # Добавляем взвешивание механизмом внимания
            if self.model_config['enable_attention']:
                # decoder_output - ([1, batch_size, dim])
                weighted_decoder_output = self.attention_module(decoder_output.squeeze(), encoder_states)
                # weighted_decoder_output - ([batch_size, dim])
                concated_attn_decoder = torch.cat([decoder_output.squeeze(), weighted_decoder_output], dim=1)
                # concated_attn_decoder - ([batch_size, 2 * dim])
                linear_vocab_proj = self.vocab_projection_layer(concated_attn_decoder)
                # concated_attn_decoder - ([batch_size, vocab_size])
            else:
                linear_vocab_proj = self.vocab_projection_layer(decoder_output)


            target_vocab_distribution = self.softmax(linear_vocab_proj)
            _, top_index = target_vocab_distribution.topk(1)
            decoder_input = top_index.reshape(1, self.batch_size, 1)

            loss += self.criterion(target_vocab_distribution.squeeze(), target_tensor[:, idx, :].squeeze())

        loss.backward()
        self.optimizer.step()
        self.optimizer_scheduler.step()

        return loss.item()

    def evaluate_batch(self, input_data, target_data):
        result_dict = dict()

        with torch.no_grad():
            encoder_output = self.encoder(input_data)

            encoder_states = encoder_output['last_hiddens']
            pooler = encoder_output['pooler']

            decoder_input = torch.tensor([[0] * self.batch_size],
                                         dtype=torch.long, device=self.device).view(1, self.batch_size, 1)
            decoder_hidden = pooler.view(1, self.batch_size, -1)

            decoder_result_list = []
            loss = 0.0
            for idx in range(self.target_tokenizer.max_sent_len):
                decoder_output, decoder_hidden = self.decoder(input_data=decoder_input,
                                                              hidden_state=decoder_hidden,
                                                              batch_size=self.batch_size)

                # Добавляем взвешивание механизмом внимания
                if self.model_config['enable_attention']:
                    # decoder_output - ([1, batch_size, dim])
                    weighted_decoder_output = self.attention_module(decoder_output.squeeze(), encoder_states)
                    # weighted_decoder_output - ([batch_size, dim])
                    concated_attn_decoder = torch.cat([decoder_output.squeeze(), weighted_decoder_output], dim=1)
                    # concated_attn_decoder - ([batch_size, 2 * dim])
                    linear_vocab_proj = self.vocab_projection_layer(concated_attn_decoder)
                    # concated_attn_decoder - ([batch_size, vocab_size])
                else:
                    linear_vocab_proj = self.vocab_projection_layer(decoder_output)

                target_vocab_distribution = self.softmax(linear_vocab_proj)
                _, top_index = target_vocab_distribution.topk(1)
                decoder_input = top_index.reshape(1, self.batch_size, 1)

                target_tensor = target_data['input_ids'].view(self.batch_size,
                                                              self.target_tokenizer.max_sent_len, 1)
                loss += self.criterion(target_vocab_distribution.squeeze(), target_tensor[:, idx, :].squeeze())
                decoder_result_list.append(list(decoder_input.flatten().cpu().numpy()))

            result_dict['loss'] = loss.item()
            decoder_result_transposed = np.array(decoder_result_list).T
            decoder_result_transposed_lists = [list(array) for array in decoder_result_transposed]

            decoded_query_list = []
            for sample in decoder_result_transposed_lists:
                decoded_query_tokens = self.target_tokenizer.decode(sample)
                query = " ".join(decoded_query_tokens)
                decoded_query_list.append(query)

            result_dict['predicted_query'] = decoded_query_list

        return result_dict
