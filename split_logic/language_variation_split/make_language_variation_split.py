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

    split_dir_saving_path = config['save_split_dir_path']
    saving_path_dir = os.path.join(os.environ['PROJECT_PATH'], split_dir_saving_path)
    if not os.path.exists(saving_path_dir):
        os.makedirs(saving_path_dir)

    language = config['language']
    assert language in ['russian', 'english']

    train_frac, dev_frac, test_frac = config['split_size']['train'], \
                                      config['split_size']['dev'], \
                                      config['split_size']['test']

    train_dataset_indexes, dev_dataset_indexes, test_dataset_indexes = split_utils.split_train_dev_test_by_indexes(
        list(range(0, len(dataset))),
        train_frac,
        dev_frac, test_frac)

    expected_keys = [split_utils.LANGUAGE2KEY_MAPPING[language], 'sparql', 'masked_sparql', 'attribute_mapping_dict',
                     'source']

    train_samples = []
    train_tokens = []
    for idx in train_dataset_indexes:
        train_sample = dataset[idx]
        new_train_sample = {split_utils.LANG_QUESTION2QUESTION_MAPPING.get(key, key): train_sample[key] for key in
                            expected_keys}
        train_samples.append(new_train_sample)
        train_tokens += new_train_sample['masked_sparql'].split()
    train_tokens_set = set(train_tokens)

    dev_samples = []
    for idx in dev_dataset_indexes:
        dev_sample = dataset[idx]
        new_dev_sample = {split_utils.LANG_QUESTION2QUESTION_MAPPING.get(key, key): dev_sample[key] for key in
                          expected_keys}
        dev_samples.append(new_dev_sample)

    test_samples = []
    for idx in test_dataset_indexes:
        test_sample = dataset[idx]
        new_test_sample = {split_utils.LANG_QUESTION2QUESTION_MAPPING.get(key, key): test_sample[key] for key in
                           expected_keys}
        test_samples.append(new_test_sample)

    cleaned_dev_samples = split_utils.check_and_clear_dataset(dataset_sample=dev_samples,
                                                              target_dataset_tokens_set=train_tokens_set)
    cleaned_test_samples = split_utils.check_and_clear_dataset(dataset_sample=test_samples,
                                                               target_dataset_tokens_set=train_tokens_set)

    print(f'Train dataset size: {len(train_samples)}')
    print(f'Dev dataset size: {len(cleaned_dev_samples)}')
    print(f'Test dataset size: {len(cleaned_test_samples)}')

    json.dump(train_samples, open(os.path.join(saving_path_dir, f'{language}_train_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(cleaned_dev_samples, open(os.path.join(saving_path_dir, f'{language}_dev_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(cleaned_test_samples, open(os.path.join(saving_path_dir, f'{language}_test_split.json'), 'w'),
              ensure_ascii=False, indent=4)

    print(f'Splits prepared and saved to {saving_path_dir} !')
