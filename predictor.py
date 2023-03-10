import os

from tqdm import tqdm

import utils
from metrics import calculate_batch_metrics


class Predictor:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def predict(self, dataloader):
        exact_match = 0
        graph_match = 0
        input_questions = []
        predicted_queries, true_queries = [], []

        self.model.eval()
        for batch in tqdm(dataloader):
            input_data, target_data = batch['nl'], batch['sparql']

            eval_result = self.model.evaluate_batch(input_data, target_data)

            pred_metrics = calculate_batch_metrics(eval_result['predicted_query'],
                                                           target_data['original_query'])
            exact_match += pred_metrics['exact_match']
            graph_match += pred_metrics['graph_match']

            input_questions += input_data['original_question']
            predicted_queries += eval_result['predicted_query']
            true_queries += target_data['original_query']

        exact_match = exact_match / len(dataloader)
        graph_match = graph_match / len(dataloader)
        result_dict = {
            "model_params": self.model.model_config,
            "exact_match_score": exact_match,
            "graph_match_score": graph_match,
            "predicted_queries": predicted_queries,
            "true_queries": true_queries,
            "input_questions": input_questions
        }

        model_dir_name, model_name = self.config["inference_model_name"].split('/')
        model_name = self.config["inference_model_name"].split('/')[-1].replace(".tar", "")
        save_preds_path = os.path.join(os.environ['PROJECT_PATH'], self.config['save_model_path'],
                                       model_dir_name,
                                       f'{model_name}_predictions.json')
        utils.save_dict(result_dict, save_preds_path)
        return result_dict