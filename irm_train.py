import torch
import argparse
import yaml
import os
import json

from target_tokenizers.query_space_tokenizer import QuerySpaceTokenizer
from irm_data_handler import IRMDataHandler
from split_logic.grammar import sparql_parser, atom_and_compound_cache
from target_tokenizers.t5_tokenizer import T5Tokenizer
from text2query_dataset import Text2QueryDataset
from trainer import Trainer
from models.irm_t5_model import IRMT5Model
from torch.utils.data import DataLoader


def run_irm(args):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'
    print('Device: ', DEVICE)

    config_name = args.config_name
    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs", config_name), 'r', encoding="utf-8")),
                       Loader=yaml.Loader)

    print('Run config: ', config)  # log to cluster logs

    model_config = config['model']
    model_name = model_config["used_model"]
    assert model_name in ['t5']
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

    sparql_parser_instance = sparql_parser.SPARQLParser(train_query_list)
    parser_with_cache = atom_and_compound_cache.AtomAndCompoundCache(sparql_parser_instance, query_key_name=None,
                                                                     return_compound_list_flag=False)
    parser_with_cache.load_cache(os.path.join(os.environ['PROJECT_PATH'], "dataset/lcquad/tmcd_split"))
    print("Loaded parser cache!")

    t5_tokenizer = T5Tokenizer(model_config=model_config)
    print('Before tokenizer vocab size: ', len(t5_tokenizer))
    prefix_tokens = [f"<{env_mask}>" for env_mask in list(sparql_parser_instance.compound_parsers_dict.keys())] + [
        '<full>']
    dataset_tokens = list(QUERY_SPACE_TOKENIZER.word2index.keys())
    # add dataset tokens + prefix tokens
    t5_tokenizer.add_tokens(prefix_tokens + dataset_tokens)
    print('After tokenizer vocab size: ', len(t5_tokenizer))

    t5_tokenizer.special_tokens_set.update(set(prefix_tokens))
    irm_data_handler = IRMDataHandler(parser=parser_with_cache, tokenizer=t5_tokenizer)

    print('Preparation train envs data')
    train_env_data = irm_data_handler.form_env_datasets(train_questions_list, train_query_list)
    print('Preparation val envs data')
    val_env_data = irm_data_handler.form_env_datasets(dev_questions_list, dev_query_list)

    train_env_dataloader_dict = {}
    for env in train_env_data:
        train_env_dataset = Text2QueryDataset(tokenized_input_list=train_env_data[env]['input'],
                                              tokenized_target_list=train_env_data[env]['target'],
                                              question_list=train_env_data[env]['question'],
                                              query_list=train_env_data[env]['query'],
                                              tokenizer=t5_tokenizer,
                                              model_type='t5',
                                              dev=DEVICE)
        train_env_dataloader = DataLoader(train_env_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        if config['test_one_batch']:
            train_env_dataloader = [list(train_env_dataloader)[0]]
        train_env_dataloader_dict[env] = train_env_dataloader

    val_env_dataloader_dict = {}
    for env in val_env_data:
        val_env_dataset = Text2QueryDataset(tokenized_input_list=val_env_data[env]['input'],
                                            tokenized_target_list=val_env_data[env]['target'],
                                            question_list=val_env_data[env]['question'],
                                            query_list=val_env_data[env]['query'],
                                            tokenizer=t5_tokenizer,
                                            model_type='t5',
                                            dev=DEVICE)
        val_env_dataloader = DataLoader(val_env_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        val_env_dataloader_dict[env] = val_env_dataloader

    t5_model = IRMT5Model(model_config=model_config, device=DEVICE, tokenizer=t5_tokenizer,
                          compound_types_list=list(train_env_data.keys()))
    trainer = Trainer(model=t5_model, config=config, model_config=model_config)

    if config['test_one_batch']:
        trainer.train_with_enviroments(train_env_dataloader_dict, train_env_dataloader_dict)
    else:
        trainer.train_with_enviroments(train_env_dataloader_dict, val_env_dataloader_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()
    run_irm(args)

