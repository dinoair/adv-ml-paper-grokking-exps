from graph_match import Graph
from utils import get_triplet_from_sparql


def execution_match(true_query, pred_query, db_id):
    # TODO: Run exec match metric
    pass


def exact_match(pred_query, true_query):
    pred_query = pred_query.lower()
    true_query = true_query.lower()
    exact_match_status = 0
    if pred_query == true_query:
        exact_match_status = 1
    return exact_match_status


def calculate_batch_metrics(pred_query_list, true_query_list):
    func_dict = {'exact_match': exact_match,
                 'graph_match': graph_match}
    result_dict = {key: 0 for key in func_dict}
    for idx in range(len(true_query_list)):
        for eval_func_name, eval_func in func_dict.items():
            result_dict[eval_func_name] += eval_func(pred_query_list[idx], true_query_list[idx])

    for key in result_dict:
        result_dict[key] /= len(true_query_list)

    return result_dict


def graph_match(pred_query, true_query):
    true_triplet = get_triplet_from_sparql(true_query)
    pred_triplet = get_triplet_from_sparql(pred_query)
    graph1 = Graph(true_triplet)
    graph2 = Graph(pred_triplet)
    return graph1.get_metric(graph2)
