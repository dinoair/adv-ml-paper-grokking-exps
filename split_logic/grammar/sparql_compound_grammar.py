from yargy import Parser
from yargy import rule, or_
from yargy.interpretation import fact
from yargy.tokenizer import TokenRule
from yargy.tokenizer import Tokenizer


class BaseGrammar:
    def __init__(self, predicates_list):
        SPACE_RULE = TokenRule('SPACE', r'\S+')
        MY_RULES = [SPACE_RULE]
        self.space_tokenizer = Tokenizer(MY_RULES)

        self.select_node = rule('select')
        self.distinct_node = rule('distinct')

        self.as_keyword = rule('as')
        self.agg_value_keyword = rule('?value')
        self.known_subj_node = rule(f'SUBJ')
        self.known_obj_node = rule(f'OBJ')
        self.unknown_subj_node = rule(f'?SUBJ')
        self.unknown_obj_node = rule(f'?OBJ')
        self.all_attrs_node = rule('*')
        self.point_node = rule('.')

        self.opening_parenthesis_node = rule('(')
        self.closing_parenthesis_node = rule(')')

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

        MASKED_VALUES = ['STR_VALUE', 'NUM_VALUE']
        self.masked_value_node = or_(*[rule(f'{mask}') for mask in MASKED_VALUES])

        self.order_keyword = rule('order').named('ORDER_KEYWORD')
        self.by_keywords = rule('by').named('BY_KEYWORD')
        self.asc_keyword = rule('asc').named('ASC')
        self.desc_keyword = rule('desc').named('DESC')


class SelectGrammar(BaseGrammar):
    def __init__(self, predicates_list):
        super(SelectGrammar, self).__init__(predicates_list)

        select_composition_fact = fact('SELECT_COMPOSITION',
                                       ['distinct_node', 'aggregation_type_node', 'project_attributes_node'])

        project_attributes_node = or_(self.unknown_subj_node,
                                      self.unknown_obj_node).interpretation(
            select_composition_fact.project_attributes_node)

        distinct_node = self.distinct_node.optional().interpretation(select_composition_fact.distinct_node)

        # (COUNT(?obj) AS ?value )
        aggregation_composition_node = rule(self.opening_parenthesis_node.optional(),
                                            self.aggregation_operator_node.interpretation(
                                                select_composition_fact.aggregation_type_node),
                                            self.opening_parenthesis_node,
                                            distinct_node,
                                            project_attributes_node,
                                            self.closing_parenthesis_node,
                                            self.as_keyword.optional(),
                                            self.agg_value_keyword.optional(),
                                            self.closing_parenthesis_node.optional()
                                            )

        project_attributes = or_(project_attributes_node, aggregation_composition_node)

        select_composition = rule(self.select_node,
                                  distinct_node,
                                  project_attributes).interpretation(select_composition_fact)

        self.parser = Parser(select_composition, tokenizer=self.space_tokenizer)


class TripletGrammar(BaseGrammar):
    def __init__(self, predicates_list):
        super(TripletGrammar, self).__init__(predicates_list)
        triplet_composition_fact = fact('TRIPLET_COMPOSITION',
                                        ['subject_node', 'predicate_node', 'object_node'])

        # TRIPLET COMPOSITION
        triplet_subj_node = or_(self.known_subj_node, self.unknown_subj_node)
        triplet_obj_node = or_(self.known_obj_node, self.unknown_obj_node)

        triplet_composition = rule(triplet_subj_node.interpretation(triplet_composition_fact.subject_node),
                                   self.predicate_node.interpretation(triplet_composition_fact.predicate_node),
                                   triplet_obj_node.interpretation(triplet_composition_fact.object_node),
                                   self.point_node.optional()).interpretation(triplet_composition_fact)

        self.parser = Parser(triplet_composition, tokenizer=self.space_tokenizer)


class FilterGrammar(BaseGrammar):
    def __init__(self, predicates_list):
        super(FilterGrammar, self).__init__(predicates_list)

        filter_composition_fact = fact('FILTER_COMPOSITION',
                                       ['string_checker_node', 'filter_function_node',
                                        'filter_argument', "comparance_operation_node",
                                        "masked_value_type_node"])

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
            filter_value_functions_node.interpretation(filter_composition_fact.filter_function_node),
            self.opening_parenthesis_node,
            filter_argument.interpretation(filter_composition_fact.filter_argument),
            self.closing_parenthesis_node
        )

        # contains( ?OBJ_3, STR_VALUE_1 ) , strstarts(lcase( ?OBJ_3 ), STR_VALUE_1 ), lang( ?OBJ_3 ) = STR_VALUE_2, ?OBJ_2 < NUM_VALUE_1
        comporator_composition = rule(
            filter_string_functions_checkers_node.optional().interpretation(
                filter_composition_fact.string_checker_node),
            self.opening_parenthesis_node.optional(),
            or_(string_function_composition, filter_argument.interpretation(filter_composition_fact.filter_argument)),
            filter_operator_node.interpretation(filter_composition_fact.comparance_operation_node),
            self.masked_value_node.interpretation(filter_composition_fact.masked_value_type_node),
            self.closing_parenthesis_node.optional()
        )

        filter_composition = rule(
            self.filter_keyword,
            self.opening_parenthesis_node,
            comporator_composition,
            self.closing_parenthesis_node,
            self.point_node.optional()
        ).interpretation(filter_composition_fact)

        self.parser = Parser(filter_composition, tokenizer=self.space_tokenizer)


class OrderGrammar(BaseGrammar):
    def __init__(self, predicates_list):
        super(OrderGrammar, self).__init__(predicates_list)

        order_composition_fact = fact('ORDER_COMPOSITION',
                                      ['sorting_type', 'order_attribute'])

        self.ordering_rule = or_(self.asc_keyword, self.desc_keyword)

        order_composition = rule(
            self.order_keyword,
            self.by_keywords,
            self.ordering_rule.interpretation(order_composition_fact.sorting_type),
            self.opening_parenthesis_node,
            or_(self.unknown_subj_node, self.unknown_obj_node).interpretation(order_composition_fact.order_attribute),
            self.closing_parenthesis_node
        ).interpretation(order_composition_fact)
        self.parser = Parser(order_composition, tokenizer=self.space_tokenizer)
