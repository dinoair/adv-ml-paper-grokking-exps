from torch import nn
import torch


class Seq2seqAttention(nn.Module):
    def __init__(self):
        super(Seq2seqAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, decoder_hidden, encoder_states):
        """
        На вход - N состояние экнодера, текущий выход декодера
        """
        decoder_hidden = decoder_hidden[:, :, None]
        decoder_to_encoder_states = torch.matmul(encoder_states, decoder_hidden)
        sm_decoder_to_encoder_states = self.softmax(decoder_to_encoder_states).transpose(2, 1)
        weighted_encoder_states = torch.matmul(sm_decoder_to_encoder_states, encoder_states).squeeze()
        return weighted_encoder_states
