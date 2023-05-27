from yargy import Parser
from yargy import rule, or_
from yargy.interpretation import fact
from yargy.tokenizer import TokenRule
from yargy.tokenizer import Tokenizer


class SparqlBaseGrammar:
    def __init__(self, predicates_list):
        SPACE_RULE = TokenRule('SPACE', r'\S+')
        MY_RULES = [SPACE_RULE]
        NUMBER_OF_ENTITIES = 5

        self.space_tokenizer = Tokenizer(MY_RULES)

        self.select_node = rule('select')
        self.ask_node = rule('ask')
        self.distinct_node = rule('distinct')

        self.where_node = rule('where')

        self.as_keyword = rule('as')
        self.agg_value_keyword = rule('?value')
        self.known_subj_node = or_(*[rule(f'SUBJ_{i+1}') for i in range(NUMBER_OF_ENTITIES)])
        self.known_obj_node = or_(*[rule(f'OBJ_{i+1}') for i in range(NUMBER_OF_ENTITIES)])
        self.unknown_subj_node = or_(*[rule(f'?SUBJ_{i+1}') for i in range(NUMBER_OF_ENTITIES)])
        self.unknown_obj_node = or_(*[rule(f'?OBJ_{i+1}') for i in range(NUMBER_OF_ENTITIES)])
        self.all_attrs_node = rule('*')
        self.point_node = rule('.')

        self.opening_parenthesis_node = rule('(')
        self.closing_parenthesis_node = rule(')')
        self.opening_brace_node = rule('{')
        self.closing_brace_node = rule('}')


        self.aggregation_operator_node = or_(*[rule('count'),
                                               rule('avg'),
                                               rule('sum'),
                                               rule('max'),
                                               rule('min')
                                               ])

        self.predicate_node = or_(*[rule(pred) for pred in predicates_list])

        self.filter_keyword = rule('filter')
        self.lang_node = rule('lang')
        self.lcase_node = rule('lcase')
        self.year_node = rule('year')

        self.contains_node = rule('contains')
        self.strstarts_node = rule('strstarts')

        self.less_node = rule('<')
        self.more_node = rule('>')
        self.equal_node = rule('=')
        self.not_equal_node = rule('!=')
        self.comma_node = rule(',')

        self.masked_str_value_node = or_(*[rule(f'STR_VALUE_{i+1}') for i in range(NUMBER_OF_ENTITIES)])
        self.masked_num_value_node = or_(*[rule(f'NUM_VALUE_{i+1}') for i in range(NUMBER_OF_ENTITIES)])
        self.masked_value_node = or_(*[self.masked_str_value_node, self.masked_num_value_node])


        self.order_keyword = rule('order').named('ORDER_KEYWORD')
        self.by_keywords = rule('by').named('BY_KEYWORD')
        self.asc_keyword = rule('asc').named('ASC')
        self.desc_keyword = rule('desc').named('DESC')


class RequestGrammarSparql(SparqlBaseGrammar):
    def __init__(self, predicates_list):
        super(RequestGrammarSparql, self).__init__(predicates_list)

        request_composition_fact = fact('request_composition',
                                       ['ask_node', 'select_composition', 'where_node'])


        project_attributes_node = or_(self.unknown_subj_node,
                                      self.unknown_obj_node)

        # (COUNT(?obj) AS ?value )
        aggregation_composition_node = rule(self.opening_parenthesis_node.optional(),
                                            self.aggregation_operator_node,
                                            self.opening_parenthesis_node,
                                            self.distinct_node.optional(),
                                            project_attributes_node,
                                            self.closing_parenthesis_node,
                                            self.as_keyword.optional(),
                                            self.agg_value_keyword.optional(),
                                            self.closing_parenthesis_node.optional()
                                            )

        select_project_attributes = or_(project_attributes_node,
                                        aggregation_composition_node)

        select_rule = rule(self.select_node,
                           self.distinct_node.optional(),
                           select_project_attributes.repeatable(),
                           self.where_node)

        ask_rule = rule(self.ask_node,
                        self.where_node)

        request_composition = or_(ask_rule.interpretation(request_composition_fact.ask_node),
                                  select_rule.interpretation(request_composition_fact.select_composition)
                                  ).interpretation(request_composition_fact)

        self.parser = Parser(request_composition, tokenizer=self.space_tokenizer)


class TripletGrammarSparql(SparqlBaseGrammar):
    def __init__(self, predicates_list):
        # TODO: вынести where в Request grammar
        super(TripletGrammarSparql, self).__init__(predicates_list)
        triplet_composition_fact = fact('triplet_composition',
                                        ['open_brace_node',
                                         'subj_node', 'predicate_node', 'obj_node',
                                         'point_node', 'close_brace_node'])

        # TRIPLET COMPOSITION
        triplet_subj_node = or_(self.known_subj_node, self.unknown_subj_node)
        triplet_obj_node = or_(self.known_obj_node, self.unknown_obj_node)

        triplet_composition = rule(self.opening_brace_node.optional().interpretation(triplet_composition_fact.open_brace_node),
                                   triplet_subj_node.interpretation(triplet_composition_fact.subj_node),
                                   self.predicate_node.interpretation(triplet_composition_fact.predicate_node),
                                   triplet_obj_node.interpretation(triplet_composition_fact.obj_node),
                                   self.point_node.optional().interpretation(triplet_composition_fact.point_node),
                                   self.closing_brace_node.optional().interpretation(triplet_composition_fact.close_brace_node)
                                   ).interpretation(triplet_composition_fact)

        self.parser = Parser(triplet_composition, tokenizer=self.space_tokenizer)


class FilterGrammarSparql(SparqlBaseGrammar):
    def __init__(self, predicates_list):
        super(FilterGrammarSparql, self).__init__(predicates_list)

        filter_composition_fact = fact('filter_composition',
                                       ['filter_keyword', 'opening_parenthesis_node',
                                        'comporator_composition', "closing_parenthesis_node",
                                        "point_node", 'closing_brace_node'])

        filter_value_functions_node = or_(self.lang_node, self.lcase_node, self.year_node)

        filter_string_functions_checkers_node = or_(self.contains_node, self.strstarts_node)
        filter_operator_node = or_(self.less_node, self.more_node, self.equal_node, self.not_equal_node,
                                   self.comma_node)

        # filter(strstarts(lcase( ?OBJ_3 ), STR_VALUE_1 ) )
        # filter(lang( ?OBJ_3 ) = STR_VALUE_2 )
        # filter ( ?OBJ_2 < NUM_VALUE_1 )
        # filter ( contains ( year ( ?OBJ_4 ) , STR_VALUE_1 ) )
        # filter(contains( ?OBJ_3, STR_VALUE_1 ) )

        filter_argument = or_(self.known_subj_node, self.unknown_subj_node, self.known_obj_node, self.unknown_obj_node)

        # lcase( ?OBJ_3 ), lang( ?OBJ_3 ), year ( ?OBJ_4 )
        string_function_composition = rule(
            filter_value_functions_node,
            self.opening_parenthesis_node,
            filter_argument,
            self.closing_parenthesis_node
        )

        # contains( ?OBJ_3, STR_VALUE_1 ) , strstarts(lcase( ?OBJ_3 ), STR_VALUE_1 ), lang( ?OBJ_3 ) = STR_VALUE_2, ?OBJ_2 < NUM_VALUE_1
        comporator_composition = rule(
            filter_string_functions_checkers_node.optional(),
            self.opening_parenthesis_node.optional(),
            or_(string_function_composition, filter_argument),
            filter_operator_node,
            self.masked_value_node,
            self.closing_parenthesis_node.optional()
        )

        filter_composition = rule(
            self.filter_keyword.interpretation(filter_composition_fact.filter_keyword),
            self.opening_parenthesis_node.interpretation(filter_composition_fact.opening_parenthesis_node),
            comporator_composition.interpretation(filter_composition_fact.comporator_composition),
            self.closing_parenthesis_node.interpretation(filter_composition_fact.closing_parenthesis_node),
            self.point_node.optional().interpretation(filter_composition_fact.point_node),
            self.closing_brace_node.optional().interpretation(filter_composition_fact.closing_brace_node),
        ).interpretation(filter_composition_fact)

        self.parser = Parser(filter_composition, tokenizer=self.space_tokenizer)


class OrderGrammarSparql(SparqlBaseGrammar):
    def __init__(self, predicates_list):
        super(OrderGrammarSparql, self).__init__(predicates_list)

        order_composition_fact = fact('order_composition',
                                      ['order_keyword', 'by_keyword',
                                       'order_rule',
                                       "opening_parenthesis_node",
                                       'order_attribute',
                                       "closing_parenthesis_node",
                                       ])

        self.ordering_rule = or_(self.asc_keyword, self.desc_keyword)

        order_composition = rule(
            self.order_keyword.interpretation(order_composition_fact.order_keyword),
            self.by_keywords.interpretation(order_composition_fact.by_keyword),
            self.ordering_rule.interpretation(order_composition_fact.order_rule),
            self.opening_parenthesis_node.interpretation(order_composition_fact.opening_parenthesis_node),
            or_(self.unknown_subj_node, self.unknown_obj_node).interpretation(order_composition_fact.order_attribute),
            self.closing_parenthesis_node.interpretation(order_composition_fact.closing_parenthesis_node)
        ).interpretation(order_composition_fact)
        self.parser = Parser(order_composition, tokenizer=self.space_tokenizer)
