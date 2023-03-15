import json
import os

import numpy as np
import yaml

import atom_and_compound_cache
import tmcd_utils
from split_logic import split_utils
from split_logic.grammar.sparql_parser import SPARQLParser

np.random.seed(42)

if __name__ == '__main__':

    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.Loader)
    split_dir_saving_path = config['save_split_dir_path']
    saving_path_dir = os.path.join(os.environ['PROJECT_PATH'], split_dir_saving_path)

    if not os.path.exists(saving_path_dir):
        os.makedirs(saving_path_dir)

    dataset_path = config['dataset_path']
    dataset = json.load(open(dataset_path, 'r', encoding='utf-8'))
    np.random.shuffle(dataset)

    chernoff_alpha_coef = config['alpha']
    language = config['language']
    assert language in ['russian', 'english']

    expected_keys = [split_utils.LANGUAGE2KEY_MAPPING[language], 'sparql', 'masked_sparql', 'attribute_mapping_dict',
                     'source']

    train_frac, dev_frac, test_frac = config['split_size']['train'], \
                                      config['split_size']['dev'], \
                                      config['split_size']['test']

    queries_list = []
    updated_dataset = []
    for sample in dataset:
        new_sample = {split_utils.LANG_QUESTION2QUESTION_MAPPING.get(key, key): sample[key] for key in expected_keys}
        queries_list.append(sample['masked_sparql'])
        new_sample['masked_sparql_no_indexes'] = SPARQLParser.preprocess_query(sample['masked_sparql'])
        updated_dataset.append(new_sample)

    sparql_compound_parser = SPARQLParser(sparql_queries_list=queries_list)
    atoms_and_compound_cache_handler = atom_and_compound_cache.AtomAndCompoundCache(parser=sparql_compound_parser,
                                                                                    query_key_name='masked_sparql_no_indexes')
    if config['load_compounds_from_file']:
        atoms_and_compound_cache_handler.load_cache(saving_path_dir)

    train_size = config['split_size']['train']
    dev_size = config['split_size']['dev']
    test_size = config['split_size']['test']

    train_samples = updated_dataset[:int(round(train_size * len(updated_dataset)))]
    test_samples = updated_dataset[int(round(train_size * len(updated_dataset))):]

    train_tokens = []
    for sample in train_samples:
        train_tokens += sample['masked_sparql'].split()
    train_tokens_set = set(train_tokens)
    cleaned_test_samples = split_utils.check_and_clear_dataset(dataset_sample=test_samples,
                                                               target_dataset_tokens_set=train_tokens_set)

    tmcd_train, tmcd_test = tmcd_utils.swap_examples(
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

    tmcd_dev = tmcd_test[:int(round(dev_size * len(updated_dataset)))]
    tmcd_test = tmcd_test[int(round(dev_size * len(updated_dataset))):]

    print(f'Train dataset size: {len(tmcd_train)}')
    print(f'Dev dataset size: {len(tmcd_dev)}')
    print(f'Test dataset size: {len(tmcd_test)}')

    json.dump(tmcd_test,
              open(os.path.join(saving_path_dir, f'{language}_train_split_coef_{chernoff_alpha_coef}.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(tmcd_dev,
              open(os.path.join(saving_path_dir, f'{language}_dev_split_coef_{chernoff_alpha_coef}.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(tmcd_test,
              open(os.path.join(saving_path_dir, f'{language}_test_split_coef_{chernoff_alpha_coef}.json'), 'w'),
              ensure_ascii=False, indent=4)
    print(f'Splits prepared and saved to {saving_path_dir} !')

    if config['save_parsed_compounds']:
        atoms_and_compound_cache_handler.dump_cache(saving_path_dir)
