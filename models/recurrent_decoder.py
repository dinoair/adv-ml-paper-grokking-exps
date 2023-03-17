import torch.nn as nn


class RecurrentDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(RecurrentDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.rnn_decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1)

    def forward(self, input_data, hidden_state, cell_state, batch_size):
        output_embeds = self.embedding(input_data).reshape(1, batch_size, self.hidden_size)
        output, (hidden_state, cell_state) = self.rnn_decoder(output_embeds, (hidden_state, cell_state))
        return output, hidden_state, cell_state