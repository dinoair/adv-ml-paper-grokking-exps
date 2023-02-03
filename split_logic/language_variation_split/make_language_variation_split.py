import os.path

import yaml
import json
import numpy as np

np.random.seed(42)

LANGUAGE2KEY_MAPPING = {
    "russian": "ru_question",
    "english": "en_question"
}

LANG_QUESTION2QUESTION_MAPPING = {
    "ru_question": "question",
    "en_question": "question",
}

def split_train_dev_test_by_indexes(index_list, train_frac, dev_frac, test_frac):
    np.random.shuffle(index_list)

    index_len = len(index_list)
    train_end_idx = int(round(index_len * train_frac))
    dev_end_idx = train_end_idx + int(round(index_len * dev_frac))
    test_end_idx = dev_end_idx + int(round(index_len * test_frac))

    train_indexes = index_list[:train_end_idx]
    dev_indexes = index_list[train_end_idx:dev_end_idx]
    test_indexes = index_list[dev_end_idx:]

    if len(dev_indexes) > len(test_indexes):
        dev_indexes = dev_indexes[:-1]
    elif len(test_indexes) > len(dev_indexes):
        test_indexes = test_indexes[:-1]

    return train_indexes, dev_indexes, test_indexes


if __name__ == "__main__":
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.Loader)

    dataset_path = config['dataset_path']
    dataset = json.load(open(dataset_path, 'r'))
    split_dir_saving_path = config['save_split_dir_path']
    saving_path_dir = os.path.join(os.environ['PROJECT_PATH'], split_dir_saving_path)


    language = config['language']
    assert language in ['russian', 'english']

    train_frac, dev_frac, test_frac = config['split_size']['train'], \
                                      config['split_size']['dev'], \
                                      config['split_size']['test']

    train_dataset_indexes, dev_dataset_indexes, test_dataset_indexes = split_train_dev_test_by_indexes(list(range(0, len(dataset))),
                                                                                                    train_frac,
                                                                                                    dev_frac, test_frac)

    expected_keys = [LANGUAGE2KEY_MAPPING[language], 'sparql', 'masked_sparql', 'attribute_mapping_dict', 'source']

    train_samples = []
    for idx in train_dataset_indexes:
        train_sample = dataset[idx]
        new_train_sample = {LANG_QUESTION2QUESTION_MAPPING.get(key, key): train_sample[key] for key in expected_keys}
        train_samples.append(new_train_sample)

    dev_samples = []
    for idx in dev_dataset_indexes:
        dev_sample = dataset[idx]
        new_dev_sample = {LANG_QUESTION2QUESTION_MAPPING.get(key, key): dev_sample[key] for key in expected_keys}
        dev_samples.append(new_dev_sample)

    test_samples = []
    for idx in test_dataset_indexes:
        test_sample = dataset[idx]
        new_test_sample = {LANG_QUESTION2QUESTION_MAPPING.get(key, key): test_sample[key] for key in expected_keys}
        test_samples.append(new_test_sample)

    print(f'Train dataset size: {len(train_samples)}')
    print(f'Dev dataset size: {len(dev_samples)}')
    print(f'Test dataset size: {len(test_samples)}')


    json.dump(train_samples, open(os.path.join(saving_path_dir, f'{language}_train_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(dev_samples, open(os.path.join(saving_path_dir, f'{language}_dev_split.json'), 'w'),
              ensure_ascii=False, indent=4)
    json.dump(test_samples, open(os.path.join(saving_path_dir, f'{language}_test_split.json'), 'w'),
              ensure_ascii=False, indent=4)

    print(f'Splits prepared and saved to {saving_path_dir} !')




