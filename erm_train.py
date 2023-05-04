import json
import os
import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.seq2seq_model import Seq2seqModel
from models.t5_model import T5Model
from target_tokenizers.query_space_tokenizer import QuerySpaceTokenizer
from target_tokenizers.t5_tokenizer import T5Tokenizer
from text2query_dataset import Text2QueryDataset
from trainer import Trainer


def run_erm(args):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'
    print('DEVICE: ', DEVICE)

    config_name = args.config_name
    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs", config_name), 'r', encoding="utf-8")),
                       Loader=yaml.Loader)

    print('Run config: ', config)# log to cluster logs

    model_config = config['model']
    model_name = model_config["used_model"]
    assert model_name in ['vanilla', 't5']
    model_config = model_config[model_name]
    batch_size = model_config['batch_size']

    train_data = json.load(
        open(os.path.join(os.environ['PROJECT_PATH'], config['data']['train']), 'r', encoding="utf-8"))
    dev_data = json.load(open(os.path.join(os.environ['PROJECT_PATH'], config['data']['dev']), 'r', encoding="utf-8"))
    dataset_vocab_path = os.path.join(os.environ['PROJECT_PATH'], config['data']['dataset_vocab'])

    train_query_list = [sample['masked_query'] for sample in train_data]
    dev_query_list = [sample['masked_query'] for sample in dev_data]

    QUERY_SPACE_TOKENIZER = QuerySpaceTokenizer(train_query_list, vocab=dataset_vocab_path, pad_flag=True)

    train_questions_list = [sample['question'] for sample in train_data]
    dev_questions_list = [sample['question'] for sample in dev_data]

    train_tokenized_questions_list = []
    dev_tokenized_questions_list = []
    train_tokenized_query_list = []
    dev_tokenized_query_list = []
    trainer = None
    target_tokenizer = None
    if model_name == 't5':
        t5_tokenizer = T5Tokenizer(model_config=model_config)
        print('Before tokenizer vocab size: ', len(t5_tokenizer))
        t5_tokenizer.add_tokens(list(QUERY_SPACE_TOKENIZER.word2index.keys()))
        print('After tokenizer vocab size: ', len(t5_tokenizer))

        train_tokenized_questions_list = t5_tokenizer(text_list=train_questions_list, max_length=512)
        dev_tokenized_questions_list = t5_tokenizer(text_list=dev_questions_list, max_length=512)
        train_tokenized_query_list = t5_tokenizer(text_list=train_query_list, max_length=128)
        dev_tokenized_query_list = t5_tokenizer(text_list=dev_query_list, max_length=128)

        t5_model = T5Model(model_config=model_config, device=DEVICE, tokenizer=t5_tokenizer)
        trainer = Trainer(model=t5_model, config=config, model_config=model_config)
        target_tokenizer = t5_tokenizer

    elif model_name == 'vanilla':
        tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])
        train_tokenized_questions_list = tokenizer(train_questions_list, padding="longest", max_length=512,
                                                   truncation=True, return_token_type_ids=True)
        dev_tokenized_questions_list = tokenizer(dev_questions_list, padding="longest", max_length=512,
                                                 truncation=True, return_token_type_ids=True)
        train_tokenized_query_list = np.array([QUERY_SPACE_TOKENIZER(query) for query in train_query_list])
        dev_tokenized_query_list = np.array([QUERY_SPACE_TOKENIZER(query) for query in dev_query_list])

        seq2seq = Seq2seqModel(model_config=model_config, device=DEVICE, target_tokenizer=QUERY_SPACE_TOKENIZER,
                               train_dataset_size=len(train_questions_list))
        trainer = Trainer(model=seq2seq, config=config, model_config=model_config)
        target_tokenizer = QUERY_SPACE_TOKENIZER

    train_dataset = Text2QueryDataset(tokenized_input_list=train_tokenized_questions_list,
                                       tokenized_target_list=train_tokenized_query_list,
                                       question_list=train_questions_list,
                                       query_list=train_query_list,
                                       tokenizer=target_tokenizer,
                                       model_type=model_name,
                                       dev=DEVICE)

    dev_dataset = Text2QueryDataset(tokenized_input_list=dev_tokenized_questions_list,
                                     tokenized_target_list=dev_tokenized_query_list,
                                     question_list=dev_questions_list,
                                     query_list=dev_query_list,
                                     tokenizer=target_tokenizer,
                                     model_type=model_name,
                                     dev=DEVICE)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


    # если хотим проверить на 1ом батче
    if config['test_one_batch']:
        train_dataloader_sample = [list(train_dataloader)[0]]
        trainer.train(train_dataloader_sample, train_dataloader_sample)
    else:
        trainer.train(train_dataloader, dev_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()
    run_erm(args)
