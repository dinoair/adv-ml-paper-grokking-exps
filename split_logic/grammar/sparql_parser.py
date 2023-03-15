import re
from collections import OrderedDict

from split_logic import split_utils
from split_logic.grammar import sparql_compound_grammar


class SPARQLParser:
    def __init__(self, sparql_queries_list):

        predicates_list = self.extract_dataset_predicates(sparql_queries_list)

        self.compound_parsers_dict = OrderedDict({
            "select_compound": sparql_compound_grammar.SelectGrammar(predicates_list).parser,
            "triplet_compound": sparql_compound_grammar.TripletGrammar(predicates_list).parser,
            "filter_compound": sparql_compound_grammar.FilterGrammar(predicates_list).parser,
            "order_compound": sparql_compound_grammar.OrderGrammar(predicates_list).parser,
        })

        self.filter_nans_lambda = lambda x: x is not None

    def extract_dataset_predicates(self, queries_list):
        predicates_list = []
        for sparql_query in queries_list:
            predicates_list += split_utils.extract_predicates(sparql_query)
        predicates_set = set(predicates_list)
        return predicates_set

    @staticmethod
    def preprocess_query(query):
        indexed_masked_values_regex = re.compile(r"(SUBJ|OBJ|NUM_VALUE|STR_VALUE)_\d+")
        tokens_list = query.split()
        updated_token_list = []
        for token in tokens_list:
            token_match = re.search(indexed_masked_values_regex, token)
            if token_match:
                cleaned_token = "_".join(token.split('_')[:-1])
                updated_token_list.append(cleaned_token)
            else:
                updated_token_list.append(token)
        updated_query = " ".join(updated_token_list).strip()
        return updated_query

    def get_atoms(self, sparql_query):
        query_tokens = set(sparql_query.split())
        return query_tokens

    def get_compounds(self, sparql_query):
        query_compound_dict = dict()
        for compound_name, compound_parser in self.compound_parsers_dict.items():
            match_list = [match for match in compound_parser.findall(sparql_query)]
            str_repr_interpretation_list = []
            if len(match_list) > 0:
                extracted_interpretation_list = [match.fact.as_json.values() for match in match_list]
                clean_interpretation_list = [list(filter(self.filter_nans_lambda, interpretation))
                                             for interpretation in extracted_interpretation_list]
                str_repr_interpretation_list = [" ".join(list(filter(self.filter_nans_lambda, interpretation)))
                                                for interpretation in clean_interpretation_list]
            query_compound_dict[compound_name] = str_repr_interpretation_list
        return query_compound_dict
