import re

LANGUAGE2KEY_MAPPING = {
    "russian": "ru_question",
    "english": "en_question"
}

LANG_QUESTION2QUESTION_MAPPING = {
    "ru_question": "question",
    "en_question": "question",
}


def extract_predicates(sparql_query):
    extract_predicates_fn = lambda x: re.findall(r'wdt:\w+|rdfs:\w+|ps:\w+|pq:\w+|p:\w+|wd:\w+', x)
    predicates_list = extract_predicates_fn(sparql_query)
    return predicates_list

def extract_dataset_predicates(queries_list):
    predicates_list = []
    for sparql_query in queries_list:
        predicates_list += extract_predicates(sparql_query)
    predicates_set = set(predicates_list)
    return predicates_set

def build_whole_vocab_set(queries_list):
    queries_tokens = set()
    for query in queries_list:
        queries_tokens.update(query.split())
    queries_tokens_set = list(queries_tokens)
    return queries_tokens_set


def split_train_dev_test_by_indexes(index_list, train_frac, dev_frac, test_frac):
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


def check_and_clear_dataset(dataset_sample, target_dataset_tokens_set, target_key_name='masked_sparql'):
    cleared_data = []
    count = 0
    for elem in dataset_sample:
        s = elem[target_key_name].split()
        has_bad_token = any([token not in target_dataset_tokens_set for token in s])
        if has_bad_token:
            count += 1
        else:
            cleared_data.append(elem)
    if count > 0:
        print('Size before clearing: ', len(dataset_sample))
        print('Deleted samples: ', count)
        print('Size after clearing: ', len(cleared_data))
    return cleared_data
