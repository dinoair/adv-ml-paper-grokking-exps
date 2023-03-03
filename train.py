import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from seq2seq_dataset import Text2SparqlDataset
from seq2seq_trainer import Seq2SeqTrainer
from sparql_tokenizer import SPARQLTokenizer
from models.seq2seq_model import Seq2seqModel
from models.t5_model import T5Model
from t5_trainer import T5Trainer
from t5_tokenizer import T5Tokenizer


def main():
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs/config.yaml"), 'r', encoding="utf-8")),
                       Loader=yaml.Loader)

    model_config = config['model']
    model_name = model_config["used_model"]
    assert model_name in ['vanilla', 't5']
    model_config = model_config[model_name]
    batch_size = model_config['batch_size']

    train_data = json.load(
        open(os.path.join(os.environ['PROJECT_PATH'], config['data']['train']), 'r', encoding="utf-8"))
    dev_data = json.load(open(os.path.join(os.environ['PROJECT_PATH'], config['data']['dev']), 'r', encoding="utf-8"))

    train_sparql_list = [sample['masked_sparql'] for sample in train_data]
    dev_sparql_list = [sample['masked_sparql'] for sample in dev_data]

    SPARQL_TOKENIZER = SPARQLTokenizer(train_sparql_list, pad_flag=True)

    train_questions_list = [sample['question'] for sample in train_data]
    dev_questions_list = [sample['question'] for sample in dev_data]

    train_tokenized_questions_list = []
    dev_tokenized_questions_list = []
    train_tokenized_sparqls_list = []
    dev_tokenized_sparqls_list = []
    trainer = None
    target_tokenizer = None
    if model_name == 't5':
        t5_tokenizer = T5Tokenizer(model_config=model_config)
        print('Before tokenizer vocab size: ', len(t5_tokenizer))
        t5_tokenizer.add_tokens(list(SPARQL_TOKENIZER.word2index.keys()))
        print('After tokenizer vocab size: ', len(t5_tokenizer))

        train_tokenized_questions_list = t5_tokenizer(text_list=train_questions_list, max_length=512)
        dev_tokenized_questions_list = t5_tokenizer(text_list=dev_questions_list, max_length=512)
        train_tokenized_sparqls_list = t5_tokenizer(text_list=train_sparql_list, max_length=128)
        dev_tokenized_sparqls_list = t5_tokenizer(dev_sparql_list, max_length=128)

        t5_model = T5Model(model_config=model_config, device=DEVICE, tokenizer=t5_tokenizer)
        trainer = T5Trainer(t5_model=t5_model, config=config, model_config=model_config)
        target_tokenizer = t5_tokenizer

    elif model_name == 'vanilla':
        tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])
        train_tokenized_questions_list = tokenizer(train_questions_list, padding="max_length",
                                                   truncation=True, return_token_type_ids=True)
        dev_tokenized_questions_list = tokenizer(dev_questions_list, padding="max_length",
                                                 truncation=True, return_token_type_ids=True)
        train_tokenized_sparqls_list = np.array([SPARQL_TOKENIZER(sparql_query) for sparql_query in train_sparql_list])
        dev_tokenized_sparqls_list = np.array([SPARQL_TOKENIZER(sparql_query) for sparql_query in dev_sparql_list])

        seq2seq = Seq2seqModel(model_config=model_config, device=DEVICE, target_tokenizer=SPARQL_TOKENIZER,
                               train_dataset_size=len(train_questions_list))
        trainer = Seq2SeqTrainer(seq2seq_model=seq2seq, config=config, model_config=model_config)
        target_tokenizer = SPARQL_TOKENIZER

    train_dataset = Text2SparqlDataset(tokenized_question_list=train_tokenized_questions_list,
                                       tokenized_sparql_list=train_tokenized_sparqls_list,
                                       question_list=train_questions_list,
                                       sparql_list=train_sparql_list,
                                       tokenizer=target_tokenizer,
                                       model_type=model_name,
                                       dev=DEVICE)

    dev_dataset = Text2SparqlDataset(tokenized_question_list=dev_tokenized_questions_list,
                                     tokenized_sparql_list=dev_tokenized_sparqls_list,
                                     question_list=dev_questions_list,
                                     sparql_list=dev_sparql_list,
                                     tokenizer=target_tokenizer,
                                     model_type=model_name,
                                     dev=DEVICE)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


    # если хотим проверить на 1ом батче
    # train_dataloader_sample = [list(train_dataloader)[0]]

    trainer.train(train_dataloader, dev_dataloader)


if __name__ == "__main__":
    main()
