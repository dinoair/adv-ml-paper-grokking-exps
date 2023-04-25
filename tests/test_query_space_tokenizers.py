import unittest
from target_tokenizers.query_space_tokenizer import QuerySpaceTokenizer


class TestQuerySpaceTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        examples = ['a ab ba', 'a ab', 'ab ab ba', 'ba ba ab a b']
        vocab_list = ['a', 'b', 'ab', 'ba', 'ca', 'cb']
        cls.space_tokenizer = QuerySpaceTokenizer(query_list=examples, vocab=vocab_list, pad_flag=True)
        cls.space_tokenizer.index2word =  {"SOS": 0, "EOS": 1, "UNK": 2, 'PAD': 3, " ": 4,
                                           'a': 5, 'b': 6, 'ab': 7, 'ba': 8, 'ca': 9, 'cb': 10}
        cls.space_tokenizer.index2word = {v:k for k, v in cls.space_tokenizer.index2word.items()}
        cls.space_tokenizer.n_words = len(cls.space_tokenizer.index2word)
        cls.space_tokenizer.max_sent_len = 7

    def test_encode_less_than_max_len(self):
        input_text = 'ba b'
        tokenized_and_padded = self.space_tokenizer(input_text)
        expected_result = [0, 8, 6, 1, 3, 3, 3]
        self.assertListEqual(tokenized_and_padded, expected_result)

    def test_encode_more_than_max_len(self):
        input_text = 'a b ba ba ab a b'
        tokenized_and_padded = self.space_tokenizer(input_text)
        expected_result = [0, 5, 6, 8, 8, 7, 1]
        self.assertListEqual(tokenized_and_padded, expected_result)