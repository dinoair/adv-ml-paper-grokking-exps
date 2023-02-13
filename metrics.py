import re

from graph_match import Graph
from utils import get_triplet_from_sparql


def exact_match(pred_sparql, true_sparql):
    pred_sparql = pred_sparql.lower()
    true_sparql = true_sparql.lower()
    exact_match_status = 0
    if pred_sparql == true_sparql:
        exact_match_status = 1
    return exact_match_status


def calculate_batch_metrics(pred_sparql_list, true_sparql_list):
    func_dict = {'exact_match': exact_match,
                 'graph_match': graph_match}
    result_dict = {key: 0 for key in func_dict}
    for idx in range(len(true_sparql_list)):
        for eval_func_name, eval_func in func_dict.items():
            result_dict[eval_func_name] += eval_func(pred_sparql_list[idx], true_sparql_list[idx])

    for key in result_dict:
        result_dict[key] /= len(true_sparql_list)

    return result_dict


def graph_match(pred_sparql, true_sparql):
    true_triplet = get_triplet_from_sparql(true_sparql)
    pred_triplet = get_triplet_from_sparql(pred_sparql)
    graph1 = Graph(true_triplet)
    graph2 = Graph(pred_triplet)
    return graph1.get_metric(graph2)
