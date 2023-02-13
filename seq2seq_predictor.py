import os

from tqdm import tqdm

import metrics
import utils

from models.seq2seq_model import Seq2seqModel


class Seq2SeqPredictor:
    def __init__(self, seq2seq_model: Seq2seqModel, config):
        self.seq2seq_model = seq2seq_model
        self.config = config
        self.batch_size = self.config['batch_size']

        self.model_save_path = os.path.join(os.environ["PROJECT_PATH"], self.config["predictions_path"])
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def predict(self, dataloader):
        exact_match = 0
        graph_match = 0
        input_questions = []
        predicted_queries, true_queries = [], []

        self.seq2seq_model.encoder.disable_bert_training()
        self.seq2seq_model.eval()
        for batch in tqdm(dataloader):
            input_data, target_data = batch['nl'], batch['sparql']

            eval_result = self.seq2seq_model.evaluate_batch(input_data, target_data)

            pred_metrics = metrics.calculate_batch_metrics(eval_result['predicted_query'],
                                                           target_data['original_query'])
            exact_match += pred_metrics['exact_match']
            graph_match += pred_metrics['graph_match']

            input_questions += input_data['original_question']
            predicted_queries += eval_result['predicted_query']
            true_queries += target_data['original_query']

        exact_match = exact_match / len(dataloader)
        graph_match = graph_match / len(dataloader)
        result_dict = {
            "exact_match_score": exact_match,
            "graph_match_score": graph_match,
            "predicted_queries": predicted_queries,
            "true_queries": true_queries,
            "input_questions": input_questions
        }
        save_preds_path = os.path.join(os.environ['PROJECT_PATH'], self.config['predictions_path'],
                                       f'{self.config["inference_model_name"].split(".")[0]}_preds.json')
        utils.save_dict(result_dict, save_preds_path)
        return result_dict
