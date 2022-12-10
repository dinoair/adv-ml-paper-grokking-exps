import unittest
import torch
from models.seq2seq_attention import Seq2seqAttention


class Seq2seqAttentionTest(unittest.TestCase):

    def test_forward_pass(self):
        batch_size, dim, encoder_states_len = 12, 32, 24

        attention_cls = Seq2seqAttention()
        decoder_hidden_state = torch.rand(batch_size, dim)
        encoder_states = torch.rand(batch_size, encoder_states_len, dim)

        attention_result = attention_cls.forward(decoder_hidden_state, encoder_states)

        expected_shape = (batch_size, dim)
        self.assertEqual(expected_shape, attention_result.shape)