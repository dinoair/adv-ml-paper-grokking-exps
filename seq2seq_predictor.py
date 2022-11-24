import torch


class Seq2SeqPredictor:
    def __init__(self, config, criterion, target_tokenizer, device):
        self.config = config
        self.device = device
        self.batch_size = self.config['batch_size']
        self.target_tokenizer = target_tokenizer
        self.criterion = criterion

    def evaluate_batch(self, encoder, decoder, input_tensor, target_data=None):
        result_dict = dict()
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            encoder_output = encoder(input_tensor)
            pooler = encoder_output['pooler']

            decoder_input = torch.tensor([[0] * self.batch_size],
                                         dtype=torch.long, device=self.device).view(1, self.batch_size, 1)
            decoder_hidden = pooler.view(1, self.batch_size, -1)
            loss = 0

            decoder_result_list = []
            for di in range(self.target_tokenizer.max_sent_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, self.batch_size)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.reshape(1, self.batch_size, 1)
                # на вход функции ошибки - распределение выхода, верные индексы на данном этапе.
                if target_data is not None:
                    target_tensor = target_data['input_ids'].view(self.batch_size,
                                                                  self.target_tokenizer.max_sent_len, 1)
                    loss += self.criterion(decoder_output.squeeze(), target_tensor[:, di, :].squeeze())
                decoder_result_list.append(list(decoder_input.flatten().cpu().numpy()))

            if target_data is not None:
                result_dict['loss'] = loss.item()

            # TODO: корявый способ обработки предсказаний для батча
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
