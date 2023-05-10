import torch
import yaml
import os
import argparse
import json

from target_tokenizers.query_space_tokenizer import QuerySpaceTokenizer
from irm_data_handler import IRMDataHandler
from split_logic.grammar import sparql_parser, atom_and_compound_cache
from target_tokenizers.t5_tokenizer import T5Tokenizer
from text2query_dataset import Text2QueryDataset
from models.irm_t5_model import IRMT5Model
from torch.utils.data import DataLoader
from predictor import Predictor


def irm_predict(args):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    config_name = args.config_name
    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs", config_name), 'r', encoding="utf-8")),
                       Loader=yaml.Loader)

    trained_model_path = os.path.join(os.environ['PROJECT_PATH'], config['save_model_path'],
                                      config['inference_model_name'].split("/")[0])

    model_config = json.load(open(os.path.join(trained_model_path, 'model_config.json'), 'r'))
    model_name = model_config["model_name"]
    assert model_name in ['t5']
    batch_size = model_config['batch_size']

    train_data = json.load(
        open(os.path.join(os.environ['PROJECT_PATH'], config['data']['train']), 'r', encoding="utf-8"))
    test_data = json.load(open(os.path.join(os.environ['PROJECT_PATH'], config['data']['test']), 'r', encoding="utf-8"))
    dataset_vocab_path = os.path.join(os.environ['PROJECT_PATH'], config['data']['dataset_vocab'])

    train_query_list = [sample['masked_query'] for sample in train_data]
    test_query_list = [sample['masked_query'] for sample in test_data]

    QUERY_SPACE_TOKENIZER = QuerySpaceTokenizer(train_query_list, vocab=dataset_vocab_path, pad_flag=True)

    train_questions_list = [sample['question'] for sample in train_data]
    test_questions_list = [sample['question'] for sample in test_data]

    sparql_parser_instance = sparql_parser.SPARQLParser(train_query_list)
    parser_with_cache = atom_and_compound_cache.AtomAndCompoundCache(sparql_parser_instance, query_key_name=None, return_compound_list_flag=False)
    parser_with_cache.load_cache(os.path.join(os.environ['PROJECT_PATH'], "dataset/lcquad/tmcd_split"))
    print("Loaded parser cache!")

    t5_tokenizer = T5Tokenizer(model_config=model_config)
    print('Before tokenizer vocab size: ', len(t5_tokenizer))
    prefix_tokens = [f"<{env_mask}>" for env_mask in list(sparql_parser_instance.compound_parsers_dict.keys())] + ['<full>']
    dataset_tokens = list(QUERY_SPACE_TOKENIZER.word2index.keys())
    # add dataset tokens + prefix tokens
    t5_tokenizer.add_tokens(prefix_tokens + dataset_tokens)
    print('After tokenizer vocab size: ', len(t5_tokenizer))

    t5_tokenizer.special_tokens_set.update(set(prefix_tokens))
    irm_data_handler = IRMDataHandler(parser=parser_with_cache, tokenizer=t5_tokenizer)

    print('Preparation train envs data')
    train_env_data = irm_data_handler.form_env_datasets(train_questions_list, train_query_list)
    print('Preparation test envs data')
    test_env_data = irm_data_handler.form_env_datasets(test_questions_list, test_query_list)

    test_full_dataset = Text2QueryDataset(tokenized_input_list=test_env_data["full"]['input'],
                                  tokenized_target_list=test_env_data["full"]['target'],
                                  question_list=test_env_data["full"]['question'],
                                  query_list=test_env_data["full"]['query'],
                                  tokenizer=t5_tokenizer,
                                  model_type='t5',
                                  dev=DEVICE)
    test_full_dataloader = DataLoader(test_full_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # we build compounds head by train data, since its what we been trained on
    irm_t5_model = IRMT5Model(model_config=model_config, device=DEVICE, tokenizer=t5_tokenizer, compound_types_list=list(train_env_data.keys()))
    t5_state = torch.load(os.path.join(os.environ['PROJECT_PATH'],
                                       config['save_model_path'], config['inference_model_name']),
                          map_location=DEVICE)['model_state_dict']
    irm_t5_model.load_state_dict(t5_state)
    predictor = Predictor(model=irm_t5_model, config=config)

    # test_dataloader_sample = [list(test_full_dataloader)[0]]

    predictor.predict(test_full_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()
    irm_predict(args)





