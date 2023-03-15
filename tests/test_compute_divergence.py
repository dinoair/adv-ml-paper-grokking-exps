import collections
import unittest

from split_logic.tmcd_split.logic import compute_divergence, _compute_new_divergence_1


class TestComputeDiv(unittest.TestCase):

    def test_compute_divergence_0(self):
        comp_0 = collections.Counter({'where { SUBJ_1 dr:political_own ?x0 . }': 73})
        comp_1 = collections.Counter({'where { SUBJ_1 dr:political_own ?x0 . }': 24})
        div = compute_divergence(comp_0, comp_1, 0.1)
        self.assertEqual(div, 0)

    def test_compute_divergence_1(self):
        comp_0 = collections.Counter({'where { SUBJ_1 dr:political_own ?x0 . }': 1})
        comp_1 = collections.Counter({'select distinct ?x0': 1})
        div = compute_divergence(comp_0, comp_1, 0.1)
        self.assertEqual(div, 1)

    def test_compute_new_divergence_0(self):
        comp_0 = collections.Counter({
            'select sum ( ?x0 )': 189,
            'select distinct ?x0': 14324,
            'where { SUBJ_1 dr:political_own ?x0 . }': 8543
        })
        comp_1 = collections.Counter({
            'select sum ( ?x0 )': 187,
            'select distinct ?x0': 14324,
            'where { SUBJ_1 dr:political_own ?x0 . }': 8543
        })
        comp_to_move = collections.Counter({
            'select sum ( ?x0 )': 1
        })
        orig_div = compute_divergence(comp_0, comp_1, 0.1)
        new_div = _compute_new_divergence_1(comp_0, comp_1, comp_to_move, orig_div, 0.1)
        self.assertEqual(new_div, 0)

    def test_compute_new_divergence_1(self):
        comp_0 = collections.Counter({
            'where { ?x0 dr:iscapitalof OBJ_1 . }': 1,
            'where { SUBJ_1 dr:mass ?x0 . }': 7,
            'where { SUBJ_1 dr:relative ?dummy . ?dummy dr:marriage_fact_now ?x0 . }': 16,
        })
        comp_1 = collections.Counter({
            'where { ?x0 dr:iscapitalof OBJ_1 . }': 19,
            'where { SUBJ_1 dr:influenced_by ?dummy . ?dummy dr:marriage_count ?x0 . }': 13,
            'where { ?x0 dr:father OBJ_1 . }': 24,
        })
        comp_to_move = collections.Counter({
            'where { ?x0 dr:iscapitalof OBJ_1 . }': 1
        })
        orig_div = compute_divergence(comp_0, comp_1, 0.1)
        new_div = _compute_new_divergence_1(comp_0, comp_1, comp_to_move, orig_div, 0.1)
        self.assertEqual(new_div, 1)