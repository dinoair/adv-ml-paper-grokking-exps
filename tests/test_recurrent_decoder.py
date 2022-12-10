import unittest
import torch

from models.recurrent_decoder import RecurrentDecoder


class RecurrentDecoderTest(unittest.TestCase):
    def test_forward_pass(self):
        decoder = RecurrentDecoder(32, 100)

        batch_size, dim = 8, 32
        decoder_input = torch.tensor([[0] * batch_size], dtype=torch.long).view(1, batch_size, 1)
        decoder_hidden = torch.rand(batch_size, dim).view(1, batch_size, -1)

        decoder_outs, decoder_hidden = decoder.forward(decoder_input, decoder_hidden, batch_size)
        self.assertEqual((1, batch_size, dim), decoder_outs.shape)  # add assertion here
