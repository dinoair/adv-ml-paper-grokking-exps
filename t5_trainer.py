from models.t5_model import T5Model
from t5_predictor import T5Predictor
from tqdm import tqdm
from metrics import calculate_batch_metrics
import utils
import os


class T5Trainer:
    def __init__(self, t5_model: T5Model, config, model_config, train_phase=True):
        self.t5_model = t5_model
        self.config = config
        self.model_config = model_config

        self.epoch_num = self.model_config['epochs_num']

        self.predictor = T5Predictor(t5_model=t5_model, config=config)
        self.save_dir_path = os.path.join(os.environ["PROJECT_PATH"], self.config['save_model_path'],
                                          self.config['run_name'])
        if train_phase:
            self.logger = utils.TXTLogger(work_dir=self.save_dir_path)

    def train(self, train_dataloader, val_dataloader):
        epoch = 0
        val_exm_epoch_acc = 0
        val_gm_epoch_acc = 0
        current_epoch_gm = 0
        try:
            for epoch in tqdm(range(self.epoch_num)):
                train_epoch_loss = 0
                for batch in train_dataloader:
                    input_data, target_data = batch['nl'], batch['sparql']
                    train_batch_loss = self.t5_model.train_on_batch(input_data, target_data)
                    train_epoch_loss += train_batch_loss
                train_epoch_loss = train_epoch_loss / len(train_dataloader)

                val_epoch_loss, val_exm_epoch_acc, val_gm_epoch_acc = 0, 0, 0
                for batch in val_dataloader:
                    input_data, target_data = batch['nl'], batch['sparql']

                    eval_result = self.t5_model.evaluate_batch(input_data=input_data, target_data=target_data)
                    val_epoch_loss += eval_result['loss']

                    val_metrics = calculate_batch_metrics(eval_result['predicted_query'], target_data['original_query'])
                    val_exm_epoch_acc += val_metrics['exact_match']
                    val_gm_epoch_acc += val_metrics['graph_match']

                val_epoch_loss = val_epoch_loss / len(val_dataloader)
                val_exm_epoch_acc = val_exm_epoch_acc / len(val_dataloader)
                val_gm_epoch_acc = val_gm_epoch_acc / len(val_dataloader)

                self.logger.log({"epoch": epoch,
                                "train_loss": train_epoch_loss,
                                "val_loss": val_epoch_loss,
                                "val_exact_match": val_exm_epoch_acc,
                                "val_graph_match": val_gm_epoch_acc,
                                "learning_rate": self.t5_model.optimizer.param_groups[0]['lr']})

                self.logger.log('********** Translation example **********')
                for input_question, true_sparql, pred_sparql in zip(input_data['original_question'][:5],
                                                                    target_data['original_query'][:5],
                                                                    eval_result['predicted_query'][:5]):
                    self.logger.log(f"NL: {input_question}")
                    self.logger.log(f"AQ: {true_sparql}")
                    self.logger.log(f"PQ: {pred_sparql}")
                    self.logger.log(" ")
                self.logger.log('******************************')

                if current_epoch_gm < val_gm_epoch_acc:
                    current_epoch_gm = val_gm_epoch_acc
                    utils.save_model(model=self.t5_model,
                                     optimizer_list=[self.t5_model.optimizer],
                                     dir_path=self.save_dir_path,
                                     filename=f"epoch_{epoch}_gm_{round(val_gm_epoch_acc, 2)}_em_{round(val_exm_epoch_acc, 2)}_t5.tar")
        except KeyboardInterrupt:
            pass

        utils.save_model(model=self.t5_model,
                         optimizer_list=[self.t5_model.optimizer],
                         dir_path=self.save_dir_path,
                         filename=f"epoch_{epoch}_gm_{round(val_gm_epoch_acc, 2)}_em_{round(val_exm_epoch_acc, 2)}_t5.tar")
        print(f'Dump model to {self.config["save_model_path"]} on {epoch} epoch!')
        print("Last val exact match: ", val_exm_epoch_acc)
        print("Last val graph match: ", val_gm_epoch_acc)