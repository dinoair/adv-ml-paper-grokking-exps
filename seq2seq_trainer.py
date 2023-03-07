import os

from tqdm import tqdm

import utils
from metrics import calculate_batch_metrics
from models.seq2seq_model import Seq2seqModel
from seq2seq_predictor import Seq2SeqPredictor
from utils import TXTLogger


class Seq2SeqTrainer:
    def __init__(self, seq2seq_model: Seq2seqModel, config, model_config, train_phase=True):
        self.seq2seq_model = seq2seq_model
        self.config = config
        self.model_config = model_config

        self.epoch_num = self.model_config['epochs_num']

        self.predictor = Seq2SeqPredictor(seq2seq_model=seq2seq_model, config=config)
        self.save_dir_path = os.path.join(os.environ["PROJECT_PATH"], self.config['save_model_path'], self.config['run_name'])
        if train_phase:
            self.logger = TXTLogger(work_dir=self.save_dir_path)
            self.logger.log(self.model_config)

    def train(self, train_dataloader, val_dataloader):
        epoch = 0
        val_exm_epoch_acc = 0
        val_gm_epoch_acc = 0
        current_epoch_gm = 0
        try:
            for epoch in tqdm(range(self.epoch_num)):
                train_epoch_loss = 0
                self.seq2seq_model.encoder.enable_bert_layers_training()
                self.seq2seq_model.train()
                for batch in train_dataloader:
                    input_data, target_data = batch['nl'], batch['sparql']
                    train_batch_loss = self.seq2seq_model.train_on_batch(input_data, target_data)
                    train_epoch_loss += train_batch_loss
                train_epoch_loss = train_epoch_loss / len(train_dataloader)

                val_epoch_loss, val_exm_epoch_acc, val_gm_epoch_acc = 0, 0, 0
                self.seq2seq_model.encoder.disable_bert_training()
                self.seq2seq_model.eval()
                for batch in val_dataloader:
                    input_data, target_data = batch['nl'], batch['sparql']

                    eval_result = self.seq2seq_model.evaluate_batch(input_data=input_data, target_data=target_data)
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
                                "learning_rate": self.seq2seq_model.optimizer_scheduler.optimizer.param_groups[0]['lr']})

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
                    utils.save_model(model=self.seq2seq_model,
                                     optimizer_list=[self.seq2seq_model.optimizer],
                                     dir_path=self.save_dir_path,
                                     filename=f"epoch_{epoch}_gm_{round(val_gm_epoch_acc, 2)}_em_{round(val_exm_epoch_acc, 2)}_seq2seq.tar")
        except KeyboardInterrupt:
            pass

        utils.save_model(model=self.seq2seq_model,
                         optimizer_list=[self.seq2seq_model.optimizer],
                         dir_path=self.save_dir_path,
                         filename=f"epoch_{epoch}_gm_{round(val_gm_epoch_acc, 2)}_em_{round(val_exm_epoch_acc, 2)}_seq2seq.tar")
        print(f'Dump model to {self.config["save_model_path"]} on {epoch} epoch!')
        print("Last val exact match: ", val_exm_epoch_acc)
        print("Last val graph match: ", val_gm_epoch_acc)
