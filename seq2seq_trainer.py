import os

import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from transformers import AutoModel

import utils
import wandb
from metrics import calculate_batch_metrics
from models.recurrent_decoder import RecurrentDecoder
from models.transformer_based_encoder import TransformerBasedEncoder
from seq2seq_predictor import Seq2SeqPredictor
from text2sparql.models.seq2seq_model import Seq2seqModel


class Seq2SeqTrainer:
    def __init__(self, seq2seq_model: Seq2seqModel, config, train_phase=True):
        self.seq2seq_model = seq2seq_model
        self.config = config

        self.batch_size = self.config['batch_size']

        self.epoch_num = self.config['epochs']

        self.predictor = Seq2SeqPredictor(seq2seq_model=seq2seq_model, config=config)

        if train_phase:
            wandb_config = {key: self.config[key] for key in ["learning_rate", "epochs",
                                                              "batch_size", "hf_transformer",
                                                              "n_last_layers2train", "batch_size", "run_name"]}

            self.wandb_run = wandb.init(project="text2sparql_language_variation", entity="oleg_oleg_96",
                                        config=wandb_config)
            self.wandb_run.name = self.config['run_name']

            self.model_save_path = os.path.join(os.environ["PROJECT_PATH"], self.config["save_model_path"])
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)

    def train(self, train_dataloader, val_dataloader):
        epoch = 0
        val_exm_epoch_acc = 0
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

                val_epoch_loss, val_exm_epoch_acc = 0, 0
                self.seq2seq_model.encoder.disable_bert_training()
                self.seq2seq_model.eval()
                for batch in val_dataloader:
                    input_data, target_data = batch['nl'], batch['sparql']

                    eval_result = self.seq2seq_model.evaluate_batch(input_data=input_data, target_data=target_data)
                    val_epoch_loss += eval_result['loss']

                    val_metrics = calculate_batch_metrics(target_data['original_query'], eval_result['predicted_query'])
                    val_exm_epoch_acc += val_metrics['exact_match']

                val_epoch_loss = val_epoch_loss / len(val_dataloader)
                val_exm_epoch_acc = val_exm_epoch_acc / len(val_dataloader)

                self.wandb_run.log({"train_loss": train_epoch_loss,
                                    "val_loss": val_epoch_loss,
                                    "val_exact_match": val_exm_epoch_acc})
        except KeyboardInterrupt:
            pass

        utils.save_model(model=self.seq2seq_model, optimizer=self.seq2seq_model.optimizer,
                         path=os.path.join(self.config['save_model_path'], f"{self.config['run_name']}_seq2seq.tar"))
        print(f'Dump model to {self.config["save_model_path"]} on {epoch} epoch!')
        print("Last val exact match: ", val_exm_epoch_acc)
        self.wandb_run.finish()
