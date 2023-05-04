
import unittest
from split_logic.grammar import sparql_parser
from irm_data_handler import IRMDataHandler

class TestIRMDataHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sparql_list = ['select distinct ?SUBJ_2 where { SUBJ_1 wdt:P2813 ?OBJ_2 . ?SUBJ_2 wdt:P31 OBJ_3 }',
                        'select ?OBJ_3 where { SUBJ_1 p:P414 ?OBJ_2 . ?SUBJ_2 ps:P414 ?OBJ_3 . ?SUBJ_2 pq:P249 ?OBJ_4 filter ( contains ( ?OBJ_4 , STR_VALUE_1 ) ) }',
                        'ask where { SUBJ_1 wdt:P106 OBJ_2 . SUBJ_1 wdt:P106 OBJ_3 }']
        parser = sparql_parser.SPARQLParser(sparql_list)
        cls.irm_data_handler = IRMDataHandler(parser=parser)

    def test_prepare_pair(self):
        input_question = 'Who is relative to Dr Dre?'
        input_query = 'select distinct ?SUBJ_2 where { SUBJ_1 wdt:P2813 ?OBJ_2 . filter ( contains ( ?OBJ_4 , STR_VALUE_1 ) ) }'
        train_data_for_query = self.irm_data_handler.prepare_pair(input_question, input_query)
        expected_result = [{
                            "model_input": "question: Who is relative to Dr Dre? query: <request> { SUBJ_1 wdt:P2813 ?OBJ_2 . filter ( contains ( ?OBJ_4 , STR_VALUE_1 ) ) }",
                            "model_output": "<request> select distinct ?SUBJ_2 where",
                            "env_name": "request"},
                           {
                           "model_input": "question: Who is relative to Dr Dre? query: select distinct ?SUBJ_2 where <triplet> filter ( contains ( ?OBJ_4 , STR_VALUE_1 ) ) }",
                           "model_output": "<triplet> { SUBJ_1 wdt:P2813 ?OBJ_2 .",
                           "env_name": "triplet"},
                            {
                            "model_input": "question: Who is relative to Dr Dre? query: select distinct ?SUBJ_2 where { SUBJ_1 wdt:P2813 ?OBJ_2 . <filter>",
                           "model_output": "<filter> filter ( contains ( ?OBJ_4 , STR_VALUE_1 ) ) }",
                            "env_name": "filter"},
                            {
                            "model_input": "question: Who is relative to Dr Dre? query: <full>",
                            "model_output": "<full> select distinct ?SUBJ_2 where { SUBJ_1 wdt:P2813 ?OBJ_2 . filter ( contains ( ?OBJ_4 , STR_VALUE_1 ) ) }",
                            "env_name": "full"}

                           ]
        self.assertListEqual(train_data_for_query, expected_result)
