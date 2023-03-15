import json
import os.path

import numpy as np
import yaml

from split_logic import split_utils


np.random.seed(42)

if __name__ == "__main__":

    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.Loader)
    dataset_path = config['dataset_path']
    dataset = json.load(open(dataset_path, 'r'))
    np.random.shuffle(dataset)

    query_vocab_set = split_utils.build_whole_vocab_set([sample['masked_sparql'] for sample in dataset])

    split_dir_saving_path = config['save_split_dir_path']
    saving_path_dir = os.path.join(os.environ['PROJECT_PATH'], split_dir_saving_path)
    if not os.path.exists(saving_path_dir):
        os.makedirs(saving_path_dir)

    language = config['language']
    assert language in ['russian', 'english']

    expected_keys = [split_utils.LANGUAGE2KEY_MAPPING[language], 'sparql', 'masked_sparql', 'attribute_mapping_dict',
                     'source']

    updated_dataset = []
    for sample in dataset:
        new_sample = {split_utils.LANG_QUESTION2QUESTION_MAPPING.get(key, key): sample[key] for key in expected_keys}
        updated_dataset.append(new_sample)

    sparql_queries_tokens_lenght_list = [len(sample['masked_sparql'].split()) for sample in updated_dataset]
    token_lenght_median = np.percentile(sparql_queries_tokens_lenght_list, 50)

    target_length_train, target_length_test = [], []
    train_tokens = []
    for sample in updated_dataset:
        sparql_queries_tokens = sample['masked_sparql'].split()
        if len(sparql_queries_tokens) <= token_lenght_median:
            target_length_train.append(sample)
        else:
            target_length_test.append(sample)

    target_length_dev = target_length_test[:len(target_length_test) // 2]
    target_length_test = target_length_test[len(target_length_test) // 2:]

    print(f'Train dataset size: {len(target_length_train)}')
    print(f'Dev dataset size: {len(target_length_dev)}')
    print(f'Test dataset size: {len(target_length_test)}')

    json.dump(target_length_train,
              open(os.path.join(saving_path_dir, f'{language}_train_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(target_length_dev,
              open(os.path.join(saving_path_dir, f'{language}_dev_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(target_length_test,
              open(os.path.join(saving_path_dir, f'{language}_test_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(query_vocab_set, open(os.path.join(os.environ['PROJECT_PATH'], f'dataset/dataset_vocab.json'), 'w'),
              ensure_ascii=False, indent=4)

    print(f'Splits prepared and saved to {saving_path_dir} !')
