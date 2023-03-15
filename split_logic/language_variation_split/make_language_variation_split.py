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

    train_frac, dev_frac, test_frac = config['split_size']['train'], \
                                      config['split_size']['dev'], \
                                      config['split_size']['test']

    train_dataset_indexes, dev_dataset_indexes, test_dataset_indexes = split_utils.split_train_dev_test_by_indexes(
        list(range(0, len(dataset))),
        train_frac,
        dev_frac, test_frac)

    expected_keys = [split_utils.LANGUAGE2KEY_MAPPING[language], 'sparql', 'masked_sparql', 'attribute_mapping_dict',
                     'source']

    updated_dataset = []
    for sample in dataset:
        new_sample = {split_utils.LANG_QUESTION2QUESTION_MAPPING.get(key, key): sample[key] for key in expected_keys}
        updated_dataset.append(new_sample)

    train_samples = [updated_dataset[idx] for idx in train_dataset_indexes]
    dev_samples = [updated_dataset[idx] for idx in dev_dataset_indexes]
    test_samples = [updated_dataset[idx] for idx in test_dataset_indexes]

    print(f'Train dataset size: {len(train_samples)}')
    print(f'Dev dataset size: {len(dev_samples)}')
    print(f'Test dataset size: {len(test_samples)}')

    json.dump(train_samples, open(os.path.join(saving_path_dir, f'{language}_train_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(dev_samples, open(os.path.join(saving_path_dir, f'{language}_dev_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(test_samples, open(os.path.join(saving_path_dir, f'{language}_test_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(query_vocab_set, open(os.path.join(os.environ['PROJECT_PATH'], f'dataset/dataset_vocab.json'), 'w'),
              ensure_ascii=False, indent=4)

    print(f'Splits prepared and saved to {saving_path_dir} !')
