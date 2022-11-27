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


class Seq2SeqTrainer:
    def __init__(self, config, device, target_tokenizer, train_phase=True):
        self.config = config
        self.device = device

        self.target_tokenizer = target_tokenizer
        self.batch_size = self.config['batch_size']

        learning_rate = self.config['learning_rate']

        hugginface_pretrained_model = self.config['hf_transformer']
        transformer_based_model = AutoModel.from_pretrained(hugginface_pretrained_model)
        trainable_layers_num = self.config['n_last_layers2train']
        self.encoder = TransformerBasedEncoder(transformer_based_model, trainable_layers_num).to(self.device)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)

        decoder_hidden_size = self.config['hidden_encoder_size']
        self.decoder = RecurrentDecoder(vocab_size=len(self.target_tokenizer.word2index),
                                        hidden_size=decoder_hidden_size).to(self.device)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        self.criterion = nn.NLLLoss()

        self.epoch_num = self.config['epochs']

        self.predictor = Seq2SeqPredictor(config=self.config, criterion=self.criterion,
                                          target_tokenizer=self.target_tokenizer, device=self.device)

        if train_phase:
            wandb_config = {key: self.config[key] for key in ["learning_rate", "hidden_encoder_size", "epochs",
                                                              "batch_size", "hf_transformer",
                                                              "n_last_layers2train", "batch_size", "run_name"]}

            self.wandb_run = wandb.init(project="text2sparql_language_variation", entity="oleg_oleg_96",
                                        config=wandb_config)
            self.wandb_run.name = self.config['run_name']

            self.model_save_path = os.path.join(os.environ["PROJECT_PATH"], self.config["save_model_path"])
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)

    def train_on_batch(self, input_data, target_data):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        self.encoder.train()
        self.decoder.train()

        encoder_output = self.encoder(input_data)
        pooler = encoder_output['pooler']

        decoder_input = torch.tensor([[0] * self.batch_size],
                                     dtype=torch.long, device=self.device).view(1, self.batch_size, 1)
        decoder_hidden = pooler.view(1, self.batch_size, -1)

        target_tensor = target_data['input_ids'].view(self.batch_size, self.target_tokenizer.max_sent_len, 1)
        target_length = target_tensor.shape[1]
        loss = 0
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, self.batch_size)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.reshape(1, self.batch_size, 1)
            # на вход функции ошибки - распределение выхода, верные индексы на данном этапе - уже юзаем reduction mean - делить на батч еще не надо
            loss += self.criterion(decoder_output.squeeze(), target_tensor[:, di, :].squeeze())

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def train(self, train_dataloader, val_dataloader):
        epoch = 0
        val_exm_epoch_acc = 0
        try:
            for epoch in tqdm(range(self.epoch_num)):
                train_epoch_loss = 0
                self.encoder.enable_bert_layers_training()
                for batch in train_dataloader:
                    input_data, target_data = batch['nl'], batch['sparql']
                    train_batch_loss = self.train_on_batch(input_data, target_data)
                    train_epoch_loss += train_batch_loss
                train_epoch_loss = train_epoch_loss / len(train_dataloader)

                val_epoch_loss, val_exm_epoch_acc = 0, 0
                self.encoder.disable_bert_training()
                for batch in val_dataloader:
                    input_data, target_data = batch['nl'], batch['sparql']

                    eval_result = self.predictor.evaluate_batch(self.encoder, self.decoder,
                                                                input_tensor=input_data, target_data=target_data)
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

        utils.save_seq2seq(self.encoder, self.encoder_optimizer, self.decoder, self.decoder_optimizer,
                           os.path.join(self.config['save_model_path'], f"{self.config['run_name']}_seq2seq.tar"))
        print(f'Dump model to {self.config["save_model_path"]} on {epoch} epoch!')
        print("Last val exact match: ", val_exm_epoch_acc)
        self.wandb_run.finish()
