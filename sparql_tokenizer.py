class SPARQLTokenizer:
    def __init__(self, language_list, pad_flag):
        self.pad_flag = pad_flag
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3: "PAD"}
        self.word2index = {"SOS": 0, "EOS": 1, "UNK": 2, 'PAD': 3}
        self.n_words = len(self.word2index)
        self.predicates_count = dict()
        self.max_sent_len = -1

        for sent in language_list:
            self.add_query(sent)
            if pad_flag:
                sent_words_amount = len(sent.split())
                if sent_words_amount > self.max_sent_len:
                    self.max_sent_len = sent_words_amount

        print(f'Tokenizer fitted - {len(self.word2index)} tokens')

    def add_query(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

            if word.startswith('dr:') or word.startswith('wdt:'):
                if word in self.predicates_count:
                    self.predicates_count[word] += 1
                else:
                    self.predicates_count[word] = 1

    def pad_sent(self, token_ids_list):
        if len(token_ids_list) < self.max_sent_len:
            padded_token_ids_list = token_ids_list + [self.word2index['PAD']] * (
                        self.max_sent_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[:self.max_sent_len - 1] + [self.word2index['EOS']]
        return padded_token_ids_list

    def __call__(self, sentence):
        tokenized_data = self.tokenize(sentence)
        if self.pad_flag:
            tokenized_data = self.pad_sent(tokenized_data)
        return tokenized_data

    def tokenize(self, sentence):
        tokenized_data = []
        tokenized_data.append(self.word2index['SOS'])
        for word in sentence.split():
            if word in self.word2index:
                tokenized_data.append(self.word2index[word])
            else:
                tokenized_data.append(self.word2index['UNK'])
        tokenized_data.append(self.word2index['EOS'])
        return tokenized_data

    def decode(self, token_list):
        predicted_tokens = []
        for token_id in token_list:
            predicted_tokens.append(self.index2word[token_id])

        return predicted_tokens