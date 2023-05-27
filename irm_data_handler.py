import copy
import random
from tqdm import tqdm

class IRMDataHandler:
    def __init__(self, cache_parser, tokenizer):
        self.cache_parser = cache_parser
        self.tokenizer = tokenizer
        self.query_envs = list(self.cache_parser.query_parser_dict.values())[0].parser_compounds + ['full']

    def prepare_pair(self, question, query, kb_id):
        query_compound_dict = self.cache_parser.get_compounds(query, kb_id)
        masked_queries_list = []
        # compound pairs
        for compound_name, compound_list in query_compound_dict.items():
            if len(compound_list) > 0:
                compound_str = random.choice(compound_list)
                mask_name = f"<{compound_name}>"
                corrupted_query = query.replace(compound_str, mask_name)
                model_input = f"question: {question} query: {corrupted_query}"
                model_output = f"{mask_name} {compound_str}"
                masked_queries_list.append({"model_input": model_input,
                                            "model_output": model_output,
                                            "env_name": compound_name,
                                            "question": question,
                                            "query": query})
        # full query
        model_input = f"question: {question} query: <full>"
        model_output = f"<full> {query}"
        masked_queries_list.append({"model_input": model_input,
                                    "model_output": model_output,
                                    "env_name": "full",
                                    "question": question,
                                    "query": query})
        return masked_queries_list

    def form_env_datasets(self, input_question_list, input_query_list, kb_id_list):
        all_pairs = []
        for question, query, kb_id in tqdm(zip(input_question_list, input_query_list, kb_id_list), total=len(input_question_list)):
            pairs_list = self.prepare_pair(question, query, kb_id)
            all_pairs += pairs_list

        # #--------------
        # import pickle
        # pickle.dump(all_pairs, open(f'/Users/somov-od/Documents/phd/projects/CompGen/text2query/dataset/lcquad/tl_all_pairs_dump_{len(all_pairs)}.pkl', 'wb'))
        # #--------------

        #collect per env data
        per_env_data_dict = {env: {"input": [], "target": [], "question": [], "query": []} for env in self.query_envs}
        per_env_data_dict['full'] = {"input": [], "target": [], "question": [], "query": []}
        for pair in all_pairs:
            env_name = pair['env_name']
            per_env_data_dict[env_name]['input'].append(pair["model_input"])
            per_env_data_dict[env_name]['target'].append(pair["model_output"])
            per_env_data_dict[env_name]['question'].append(pair['question'])
            per_env_data_dict[env_name]['query'].append(pair['query'])

        env_data_dict = dict()
        for env_name in per_env_data_dict:
            if len(per_env_data_dict[env_name]['input']) > 0:
                env_data_dict[env_name] = copy.deepcopy(per_env_data_dict[env_name])
                env_data_dict[env_name]['input'] = self.tokenizer(per_env_data_dict[env_name]['input'], max_length=512)
                env_data_dict[env_name]['target'] = self.tokenizer(per_env_data_dict[env_name]['target'], max_length=128)

        return env_data_dict



