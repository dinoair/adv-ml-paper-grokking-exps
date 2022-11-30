import os

import torch
from tqdm import tqdm

import metrics
import utils


class Seq2SeqPredictor:
    def __init__(self, config, criterion, target_tokenizer, device):
        self.config = config
        self.device = device
        self.batch_size = self.config['batch_size']
        self.target_tokenizer = target_tokenizer
        self.criterion = criterion

        self.model_save_path = os.path.join(os.environ["PROJECT_PATH"], self.config["predictions_path"])
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

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

    def predict(self, encoder, decoder, dataloader):
        exact_match = 0
        input_questions = []
        predicted_queries, true_queries = [], []

        encoder.disable_bert_training()

        for batch in tqdm(dataloader):
            input_data, target_data = batch['nl'], batch['sparql']

            eval_result = self.evaluate_batch(encoder, decoder, input_data, target_data)

            pred_metrics = metrics.calculate_batch_metrics(target_data['original_query'], eval_result['predicted_query'])
            exact_match += pred_metrics['exact_match']

            input_questions += input_data['original_question']
            predicted_queries += eval_result['predicted_query']
            true_queries += target_data['original_query']

        exact_match = exact_match / len(dataloader)
        result_dict = {

            "exact_match_score": exact_match,
            "predicted_queries": predicted_queries,
            "true_queries": true_queries,
            "input_questions": input_questions
        }
        save_preds_path = os.path.join(os.environ['PROJECT_PATH'], self.config['predictions_path'],
                                       f'{self.config["inference_model_name"].split(".")[0]}_preds.json')
        utils.save_dict(result_dict, save_preds_path)
        return result_dict

