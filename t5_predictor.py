from models.t5_model import T5Model
import os
from tqdm import tqdm
from metrics import calculate_batch_metrics
import utils

class T5Predictor:
    def __init__(self, t5_model: T5Model, config):
        self.t5_model = t5_model
        self.config = config

        self.model_save_path = os.path.join(os.environ["PROJECT_PATH"], self.config["predictions_path"])
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def predict(self, dataloader):
        exact_match = 0
        graph_match = 0
        input_questions = []
        predicted_queries, true_queries = [], []

        for batch in tqdm(dataloader):
            input_data, target_data = batch['nl'], batch['sparql']

            eval_result = self.t5_model.evaluate_batch(input_data, target_data)

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