from yargy import Parser
from yargy import rule, or_
from yargy.interpretation import fact
from yargy.tokenizer import TokenRule
from yargy.tokenizer import Tokenizer


class SQLBaseGrammar:
    def __init__(self, db_attribute_list):
        SPACE_RULE = TokenRule('SPACE', r'\S+')
        MY_RULES = [SPACE_RULE]
        NUMBER_OF_ENTITIES = 5

        self.space_tokenizer = Tokenizer(MY_RULES)

        self.select_node = rule('SELECT')
        self.from_node = rule('FROM')
        self.table_node = rule('table')

        self.where_node = rule('WHERE')

        self.less_node = rule('<')
        self.more_node = rule('>')
        self.equal_node = rule('=')
        self.not_equal_node = rule('!=')

        self.count_node = rule('COUNT')
        self.min_node = rule('MIN')
        self.max_node = rule('MAX')
        self.sum_node = rule('SUM')
        self.avg_node = rule('AVG')

        self.masked_str_value_node = or_(*[rule(f'STR_VALUE_{i + 1}') for i in range(NUMBER_OF_ENTITIES)])
        self.masked_num_value_node = or_(*[rule(f'NUM_VALUE_{i + 1}') for i in range(NUMBER_OF_ENTITIES)])

        self.and_node = rule('AND')
        self.or_node = rule('OR')

        self.attribute_node = or_(*[rule(attr) for attr in db_attribute_list])


class RequestGrammarSql(SQLBaseGrammar):
    def __init__(self, db_attribute_list):
        super(RequestGrammarSql, self).__init__(db_attribute_list)

        request_composition_fact = fact('request_composition',
                                        ['select_composition', 'from_node', 'table_node', 'where_node'])

        aggregation_node = or_(self.count_node, self.min_node, self.max_node,
                               self.avg_node, self.sum_node)

        select_composition_node = rule(self.select_node,
                                       aggregation_node.optional(),
                                       self.attribute_node)

        request_composition = rule(select_composition_node.interpretation(request_composition_fact.select_composition),
                                   self.from_node.interpretation(request_composition_fact.from_node),
                                   self.table_node.interpretation(request_composition_fact.table_node),
                                   self.where_node.optional().interpretation(request_composition_fact.where_node)).interpretation(request_composition_fact)

        self.parser = Parser(request_composition, tokenizer=self.space_tokenizer)

class ConditionGrammarSql(SQLBaseGrammar):
    def __init__(self, db_attribute_list):
        super(ConditionGrammarSql, self).__init__(db_attribute_list)

        condition_composition_fact = fact('condition_composition',
                                          ['attribute_node', 'comparison_node',
                                           'comparison_node_value', 'and_node'])

        comparison_node = or_(self.less_node, self.more_node, self.equal_node,
                              self.max_node, self.sum_node, self.avg_node)
        comparison_value_node = or_(self.masked_str_value_node, self.masked_num_value_node)

        condition_composition = rule(self.attribute_node.interpretation(condition_composition_fact.attribute_node),
                                     comparison_node.interpretation(condition_composition_fact.comparison_node),
                                     comparison_value_node.interpretation(condition_composition_fact.comparison_node_value),
                                     self.and_node.optional().interpretation(condition_composition_fact.and_node)).interpretation(condition_composition_fact)

        self.parser = Parser(condition_composition, tokenizer=self.space_tokenizer)

