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


def main():
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs/config.yaml"), 'r', encoding="utf-8")), Loader=yaml.Loader)

    tokenizer_name = config['hf_tokenizer']
    RU_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)

    train_data = json.load(open(os.path.join(os.environ['PROJECT_PATH'], config['data']['train']), 'r', encoding="utf-8"))
    dev_data = json.load(open(os.path.join(os.environ['PROJECT_PATH'], config['data']['dev']), 'r', encoding="utf-8"))

    train_sparql_list = [sample['masked_sparql_query'] for sample in train_data]
    dev_sparql_list = [sample['masked_sparql_query'] for sample in dev_data]
    SPARQL_TOKENIZER = SPARQLTokenizer(train_sparql_list, pad_flag=True)

    train_tokenized_sparqls_list = np.array([SPARQL_TOKENIZER(sparql_query) for sparql_query in train_sparql_list])
    dev_tokenized_sparqls_list = np.array([SPARQL_TOKENIZER(sparql_query) for sparql_query in dev_sparql_list])

    train_questions_list = [sample['filled_paraphrased_canonical'] for sample in train_data]
    dev_questions_list = [sample['filled_paraphrased_canonical'] for sample in dev_data]

    train_tokenized_questions_list = RU_TOKENIZER(train_questions_list, return_tensors="pt", padding=True,
                                                  truncation=True)
    dev_tokenized_questions_list = RU_TOKENIZER(dev_questions_list, return_tensors="pt", padding=True, truncation=True)

    train_dataset = Text2SparqlDataset(tokenized_question_list=train_tokenized_questions_list,
                                       tokenized_sparql_list=train_tokenized_sparqls_list,
                                       question_list=train_questions_list,
                                       sparql_list=train_sparql_list,
                                       dev=DEVICE)
    
    dev_dataset = Text2SparqlDataset(tokenized_question_list=dev_tokenized_questions_list,
                                   tokenized_sparql_list=dev_tokenized_sparqls_list,
                                   question_list=dev_questions_list,
                                   sparql_list=dev_sparql_list,
                                   dev=DEVICE)
    
    batch_size = config['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    seq2seq_model = Seq2seqModel(config=config, device=DEVICE, target_tokenizer=SPARQL_TOKENIZER)

    trainer = Seq2SeqTrainer(seq2seq_model=seq2seq_model, config=config)
    
    # если хотим проверить на 1ом батче
    train_dataloader_sample = [list(train_dataloader)[0]]
    
    trainer.train(train_dataloader_sample, train_dataloader_sample)


if __name__ == "__main__":
    main()