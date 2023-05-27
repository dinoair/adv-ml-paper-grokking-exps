from collections import OrderedDict

from split_logic.grammar import sql_compound_grammar


class SQLParser:
    def __init__(self, db_attribute_list):

        self.compound_parsers_dict = OrderedDict({
            "request": sql_compound_grammar.RequestGrammarSql(db_attribute_list).parser,
            "condition": sql_compound_grammar.ConditionGrammarSql(db_attribute_list).parser,
        })
        self.parser_compounds = list(self.compound_parsers_dict.keys())
        self.filter_nans_lambda = lambda x: x is not None

    def get_atoms(self, sql_query):
        query_tokens = set(sql_query.split())
        return query_tokens

    def get_compounds(self, sql_query):
        query_compound_dict = dict()
        for compound_name, compound_parser in self.compound_parsers_dict.items():
            match_list = [match for match in compound_parser.findall(sql_query)]
            str_repr_interpretation_list = []
            if len(match_list) > 0:
                extracted_interpretation_list = [match.fact.as_json.values() for match in match_list]
                clean_interpretation_list = [list(filter(self.filter_nans_lambda, interpretation))
                                             for interpretation in extracted_interpretation_list]
                str_repr_interpretation_list = [" ".join(list(filter(self.filter_nans_lambda, interpretation)))
                                                for interpretation in clean_interpretation_list]
            query_compound_dict[compound_name] = str_repr_interpretation_list
        return query_compound_dict
