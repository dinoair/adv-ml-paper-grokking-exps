import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.seq2seq_model import Seq2seqModel
from models.t5_model import T5Model
from predictor import Predictor
from sparql_tokenizer import SPARQLTokenizer
from t5_tokenizer import T5Tokenizer
from text2sparql_dataset import Text2SparqlDataset


def main():
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'


    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs/config.yaml"), 'r', encoding="utf-8")),
                       Loader=yaml.Loader)
    trained_model_path = os.path.join(os.environ['PROJECT_PATH'], config['save_model_path'],
                                      config['inference_model_name'].split("/")[0])

    model_config = json.load(open(os.path.join(trained_model_path, 'model_config.json'), 'r'))
    model_name = model_config["model_name"]
    assert model_name in ['vanilla', 't5']
    batch_size = model_config['batch_size']

    train_data = json.load(
        open(os.path.join(os.environ['PROJECT_PATH'], config['data']['train']), 'r', encoding="utf-8"))
    test_data = json.load(
        open(os.path.join(os.environ['PROJECT_PATH'], config['data']['test']), 'r', encoding="utf-8"))
    dataset_vocab_path = os.path.join(os.environ['PROJECT_PATH'], config['data']['dataset_vocab'])

    train_sparql_list = [sample['masked_sparql'] for sample in train_data]
    test_sparql_list = [sample['masked_sparql'] for sample in test_data]

    SPARQL_TOKENIZER = SPARQLTokenizer(train_sparql_list, vocab_path=dataset_vocab_path, pad_flag=True)

    train_questions_list = [sample['question'] for sample in train_data]
    test_questions_list = [sample['question'] for sample in test_data]

    test_tokenized_questions_list = []
    test_tokenized_sparqls_list = []
    predictor = None
    target_tokenizer = None
    if model_name == 't5':
        t5_tokenizer = T5Tokenizer(model_config=model_config)
        print('Before tokenizer vocab size: ', len(t5_tokenizer))
        t5_tokenizer.add_tokens(list(SPARQL_TOKENIZER.word2index.keys()))
        print('After tokenizer vocab size: ', len(t5_tokenizer))

        test_tokenized_questions_list = t5_tokenizer(text_list=test_questions_list, max_length=512)
        test_tokenized_sparqls_list = t5_tokenizer(text_list=test_sparql_list, max_length=128)
        t5_state = torch.load(os.path.join(os.environ['PROJECT_PATH'],
                                           config['save_model_path'], config['inference_model_name']),
                              map_location=DEVICE)['model_state_dict']
        t5_model = T5Model(model_config=model_config, device=DEVICE, tokenizer=t5_tokenizer)
        t5_model.load_state_dict(t5_state)
        predictor = Predictor(model=t5_model, config=config)
        target_tokenizer = t5_tokenizer

    elif model_name == 'vanilla':
        tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])
        test_tokenized_questions_list = tokenizer(test_questions_list, padding="longest", max_length=512,
                                                  truncation=True, return_token_type_ids=True)
        test_tokenized_sparqls_list = np.array([SPARQL_TOKENIZER(sparql_query) for sparql_query in test_sparql_list])

        seq2seq_state = torch.load(os.path.join(os.environ['PROJECT_PATH'],
                                                config['save_model_path'], config['inference_model_name']),
                                   map_location=DEVICE)['model_state_dict']
        seq2seq = Seq2seqModel(model_config=model_config, device=DEVICE, target_tokenizer=SPARQL_TOKENIZER,
                               train_dataset_size=len(train_questions_list))
        seq2seq.load_state_dict(seq2seq_state)
        predictor = Predictor(model=seq2seq, config=config)
        target_tokenizer = SPARQL_TOKENIZER

    test_dataset = Text2SparqlDataset(tokenized_question_list=test_tokenized_questions_list,
                                      tokenized_sparql_list=test_tokenized_sparqls_list,
                                      question_list=test_questions_list,
                                      sparql_list=test_sparql_list,
                                      tokenizer=target_tokenizer,
                                      model_type=model_name,
                                      dev=DEVICE)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # test_dataloader_sample = [list(test_dataloader)[0]]

    predictor.predict(test_dataloader)


if __name__ == "__main__":
    main()
