import json
import os

import numpy as np
import yaml

import tmcd_utils
from split_logic import split_utils
from split_logic.grammar import sparql_parser, atom_and_compound_cache, sql_parser

np.random.seed(42)

if __name__ == '__main__':
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.Loader)
    split_dir_saving_path = config['save_split_dir_path']
    dataset_dir_path = os.path.dirname(split_dir_saving_path)
    saving_path_dir = os.path.join(os.environ['PROJECT_PATH'], split_dir_saving_path)

    if not os.path.exists(saving_path_dir):
        os.makedirs(saving_path_dir)

    dataset_path = config['dataset_path']
    dataset = json.load(open(dataset_path, 'r', encoding='utf-8'))
    np.random.shuffle(dataset)

    query_vocab_set = split_utils.build_whole_vocab_set([sample['masked_query'] for sample in dataset])

    chernoff_alpha_coef = config['alpha']
    language = config['language']
    assert language in ['russian', 'english']

    expected_keys = [split_utils.LANGUAGE2KEY_MAPPING[language], 'query', 'masked_query', 'attribute_mapping_dict',
                     'source', 'kb_id']

    queries_list = []
    updated_dataset = []
    for sample in dataset:
        new_sample = {split_utils.LANG_QUESTION2QUESTION_MAPPING.get(key, key): sample[key] for key in expected_keys}
        queries_list.append(sample['masked_query'])
        updated_dataset.append(new_sample)

    parser_instance = None
    parser_dict = dict()
    # TODO: Add SQL TMCD run
    if config['query_language'] == 'sparql':
        parser_instance = sparql_parser.SPARQLParser(sparql_queries_list=queries_list)
        parser_dict = {'wikidata': parser_instance}
    atoms_and_compound_cache_handler = atom_and_compound_cache.AtomAndCompoundCache(parser_dict=parser_dict,
                                                                                    query_key_name='masked_query',
                                                                                    kb_id_key_name='kb_id',
                                                                                    return_compound_list_flag=True,
                                                                                    compound_cache_path=None)

    train_samples = updated_dataset[:len(updated_dataset) // 2]
    test_samples = updated_dataset[len(updated_dataset) // 2:]

    train_samples, test_samples = tmcd_utils.swap_examples(
        train_samples,
        test_samples,
        get_compounds_fn=atoms_and_compound_cache_handler.get_compounds,
        get_atoms_fn=atoms_and_compound_cache_handler.get_atoms,
        max_iterations=100000,
        max_divergence=1,
        min_atom_count=1,
        print_frequencies=False,
        coef=chernoff_alpha_coef
    )

    train_tokens = []
    for sample in train_samples:
        train_tokens += sample['masked_query'].split()
    train_tokens_set = set(train_tokens)

    cleaned_test_samples = split_utils.align_test_dataset_with_train_tokens(test_samples,
                                                                            target_dataset_tokens_set=train_tokens_set,
                                                                            target_key_name='masked_query')

    dev_samples = test_samples[:len(cleaned_test_samples) // 2]
    test_samples = test_samples[len(cleaned_test_samples) // 2:]

    print(f'Train dataset size: {len(train_samples)}')
    print(f'Dev dataset size: {len(dev_samples)}')
    print(f'Test dataset size: {len(test_samples)}')

    json.dump(train_samples,
              open(os.path.join(saving_path_dir, f'{language}_train_split_coef_{chernoff_alpha_coef}.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(dev_samples,
              open(os.path.join(saving_path_dir, f'{language}_dev_split_coef_{chernoff_alpha_coef}.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(test_samples,
              open(os.path.join(saving_path_dir, f'{language}_test_split_coef_{chernoff_alpha_coef}.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(query_vocab_set, open(os.path.join(os.environ['PROJECT_PATH'], f'{dataset_dir_path}/query_vocab.json'), 'w'),
              ensure_ascii=False, indent=4)

    print(f'Splits prepared and saved to {saving_path_dir} !')

    if config['save_parsed_compounds']:
        atoms_and_compound_cache_handler.dump_cache(saving_path_dir)
