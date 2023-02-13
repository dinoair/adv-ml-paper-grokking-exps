import unittest

from metrics import graph_match


class MetricTest(unittest.TestCase):
    def test_graph_match(self):
        text1 = '{ subj_1 pred_1 obj_1 }'
        text2 = '{ subj_1 pred_1 obj_1 }'
        self.assertEqual(graph_match(text1, text2), 1.0)

        text1 = '{ subj_1 pred_1 obj_1 . subj_2 pred_2 obj_2 }'
        text2 = '{ subj_1 pred_1 obj_1 . subj_3 pred_3 obj_3 }'
        self.assertEqual(graph_match(text1, text2), 0.5)

        text1 = '{ subj_1 pred_1 obj_1 . subj_2 pred_2 obj_2 }'
        text2 = '{ subj_1 pred_1 obj_1 . subj_3 pred_3 obj_2 }'
        self.assertEqual(graph_match(text1, text2), 0.5)

        text1 = '{ subj_1 pred_1 obj_1 . subj_2 pred_2 obj_2 }'
        text2 = '{ subj_1 pred_1 obj_1 . subj_2 pred_3 obj_2 }'
        self.assertEqual(graph_match(text1, text2), 0.5)

        text1 = '{ subj_1 pred_1 obj_1 . subj_1 pred_2 obj_2 }'
        text2 = '{ subj_1 pred_1 obj_1 . subj_1 pred_3 obj_2 }'
        self.assertEqual(graph_match(text1, text2), 0.6)

        text1 = '{ subj_1 pred_1 obj_1 . obj_1 pred_2 obj_2 }'
        text2 = '{ subj_1 pred_1 obj_1 . obj_1 pred_3 obj_2 }'
        self.assertEqual(graph_match(text1, text2), 0.6)

        text1 = '{ subj_1 pred_1 obj_1 . obj_1 pred_2 obj_2 }'
        text2 = '{ obj_1 pred_2 obj_2 . subj_1 pred_1 obj_1 }'
        self.assertEqual(graph_match(text1, text2), 1.0)

