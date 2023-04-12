import json
import os.path

import numpy as np
import yaml

from split_logic import split_utils

np.random.seed(42)

if __name__ == "__main__":

    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.Loader)
    dataset_path = config['dataset_path']
    split_dir_saving_path = config['save_split_dir_path']
    dataset_dir_path = os.path.dirname(split_dir_saving_path)
    dataset = json.load(open(dataset_path, 'r'))
    np.random.shuffle(dataset)

    query_vocab_set = split_utils.build_whole_vocab_set([sample['masked_query'] for sample in dataset])

    saving_path_dir = os.path.join(os.environ['PROJECT_PATH'], split_dir_saving_path)
    if not os.path.exists(saving_path_dir):
        os.makedirs(saving_path_dir)

    language = config['language']
    assert language in ['russian', 'english']

    expected_keys = [split_utils.LANGUAGE2KEY_MAPPING[language], 'query', 'masked_query', 'attribute_mapping_dict',
                     'source']

    updated_dataset = []
    for sample in dataset:
        new_sample = {split_utils.LANG_QUESTION2QUESTION_MAPPING.get(key, key): sample[key] for key in expected_keys}
        updated_dataset.append(new_sample)

    query_tokens_lenght_list = [len(sample['masked_query'].split()) for sample in updated_dataset]
    token_lenght_percentile = np.percentile(query_tokens_lenght_list, config['train_percentile'])

    # Короткие и длинные запросы могут не пересекаться по токенам,
    # так как в длинных запросах используются новые предикаты или субъекты/объекты
    # Поэтому в давайте в тесте оставим короткие запросы, а в трейне длинные запросы
    train_samples, test_samples = [], []
    train_tokens = []
    train_tokens_set = set()
    for sample in updated_dataset:
        query_tokens = sample['masked_query'].split()
        if len(query_tokens) <= token_lenght_percentile:
            test_samples.append(sample)
        else:
            train_samples.append(sample)
            train_tokens_set = train_tokens_set.union(query_tokens)

    cleaned_test_samples = split_utils.align_test_dataset_with_train_tokens(test_samples,
                                                                            target_dataset_tokens_set=train_tokens_set,
                                                                            target_key_name='masked_query')

    dev_samples = cleaned_test_samples[:len(cleaned_test_samples) // 2]
    test_samples = cleaned_test_samples[len(cleaned_test_samples) // 2:]

    print(f'Train dataset size: {len(train_samples)}')
    print(f'Dev dataset size: {len(dev_samples)}')
    print(f'Test dataset size: {len(test_samples)}')

    json.dump(train_samples,
              open(os.path.join(saving_path_dir, f'{language}_train_split_above_{config["train_percentile"]}_percentile.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(dev_samples,
              open(os.path.join(saving_path_dir, f'{language}_dev_split_below_{config["train_percentile"]}_percentile.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(test_samples,
              open(os.path.join(saving_path_dir, f'{language}_test_split_below_{config["train_percentile"]}_percentile.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(query_vocab_set,
              open(os.path.join(os.environ['PROJECT_PATH'], f'{dataset_dir_path}/query_vocab.json'), 'w'),
              ensure_ascii=False, indent=4)

    print(f'Splits prepared and saved to {saving_path_dir} !')
