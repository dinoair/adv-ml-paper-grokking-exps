import yaml
import json
import os
import torch
import numpy as np
from seq2seq_trainer import Seq2SeqTrainer
from transformers import AutoTokenizer
from seq2seq_dataset import Text2SparqlDataset
from sparql_tokenizer import SPARQLTokenizer
from torch.utils.data import DataLoader


def main():
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs/config.yaml"), 'r')), Loader=yaml.Loader)

    tokenizer_name = config['hf_tokenizer']
    RU_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)

    train_data = json.load(open(config['data']['train'], 'r'))
    dev_data = json.load(open(config['data']['dev'], 'r'))

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
    
    train_dataloader_sample = [list(train_dataloader)[0]]

    trainer = Seq2SeqTrainer(config=config, device=DEVICE, target_tokenizer=SPARQL_TOKENIZER)

    trainer.train(train_dataloader_sample, train_dataloader_sample)


if __name__ == "__main__":
    main()