import torch
from torch import nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel

from models.recurrent_decoder import RecurrentDecoder
from models.seq2seq_attention import Seq2seqAttention
from models.transformer_based_encoder import TransformerBasedEncoder
from sparql_tokenizer import SPARQLTokenizer


class Seq2seqModel(nn.Module):
    def __init__(self, config: dict, device: str, target_tokenizer: SPARQLTokenizer):
        super(Seq2seqModel, self).__init__()
        self.device = device
        self.target_tokenizer = target_tokenizer
        self.batch_size = config['batch_size']

        hugginface_pretrained_model = config['hf_transformer']
        transformer_based_model = AutoModel.from_pretrained(hugginface_pretrained_model)
        trainable_layers_num = config['n_last_layers2train']
        self.encoder: TransformerBasedEncoder = TransformerBasedEncoder(transformer_based_model,
                                                                        trainable_layers_num).to(self.device)

        self.attention_module = Seq2seqAttention().to(self.device)

        decoder_hidden_state = self.encoder.bert_module.pooler.dense.weight.shape[0]
        self.decoder: RecurrentDecoder = RecurrentDecoder(hidden_size=decoder_hidden_state,
                                                          vocab_size=len(target_tokenizer.word2index)).to(self.device)
        self.attention_head = nn.Linear(2 * decoder_hidden_state, len(target_tokenizer.word2index)).to(self.device)

        self.softmax = nn.LogSoftmax(dim=1).to(self.device)

        self.optimizer = AdamW([{"params": self.encoder.parameters()},
                                {"params": self.attention_module.parameters()},
                                {"params": self.decoder.parameters()},
                                {"params": self.attention_head.parameters()}])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, config['num_warmup_steps'],
                                                         self.batch_size * config['epochs'])
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
            # decoder_output - ([1, batch_size, dim])

            weighted_decoder_output = self.attention_module(decoder_output.squeeze(), encoder_states)
            # weighted_decoder_output - ([batch_size, dim])
            concated_attn_decoder = torch.cat([decoder_output.squeeze(), weighted_decoder_output], dim=1)
            # concated_attn_decoder - ([batch_size, 2 * dim])
            linear_vocab_proj = self.attention_head(concated_attn_decoder)
            # concated_attn_decoder - ([batch_size, vocab_size])
            target_vocab_distribution = self.softmax(linear_vocab_proj)
            # concated_attn_decoder - ([batch_size, vocab_size])

            _, top_index = target_vocab_distribution.topk(1)
            decoder_input = top_index.reshape(1, self.batch_size, 1)

            loss += self.criterion(target_vocab_distribution.squeeze(), target_tensor[:, idx, :].squeeze())

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def evaluate_batch(self, input_data, target_data=None):
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

                weighted_decoder_output = self.attention_module(decoder_output.squeeze(), encoder_states)
                # weighted_decoder_output - ([batch_size, dim])
                concated_attn_decoder = torch.cat([decoder_output.squeeze(), weighted_decoder_output], dim=1)
                # concated_attn_decoder - ([batch_size, 2 * dim])
                linear_vocab_proj = self.attention_head(concated_attn_decoder)
                # concated_attn_decoder - ([batch_size, vocab_size])
                target_vocab_distribution = self.softmax(linear_vocab_proj)
                # concated_attn_decoder - ([batch_size, vocab_size])

                _, top_index = target_vocab_distribution.topk(1)
                decoder_input = top_index.reshape(1, self.batch_size, 1)

                if target_data is not None:
                    target_tensor = target_data['input_ids'].view(self.batch_size,
                                                                  self.target_tokenizer.max_sent_len, 1)
                    loss += self.criterion(target_vocab_distribution.squeeze(), target_tensor[:, idx, :].squeeze())
                decoder_result_list.append(list(decoder_input.flatten().cpu().numpy()))

            if target_data is not None:
                result_dict['loss'] = loss.item()

            batch_preds_list = [[] for _ in range(self.batch_size)]
            for batch in decoder_result_list:
                for idx, sample_idx in enumerate(batch):
                    query_token = self.target_tokenizer.index2word[sample_idx]
                    batch_preds_list[idx].append(query_token)

            for idx in range(self.batch_size):
                filtered_sample = list(filter(lambda x: x not in ['SOS', 'EOS', 'PAD'], batch_preds_list[idx]))
                query = " ".join(filtered_sample)
                batch_preds_list[idx] = query

            result_dict['predicted_query'] = batch_preds_list

        return result_dict
