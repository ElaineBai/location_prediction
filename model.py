import torch
import torch.nn as nn
from torch.autograd import *

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layer, output_size):
        super(LSTM, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.rnn = nn.LSTM(input_size, hidden_dim, num_layer)
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, num_layer):
        return (Variable(torch.zeros(num_layer, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(num_layer, 1, self.hidden_dim)).cuda())

    def forward(self, seq):
        lstm_out, self.hidden = self.rnn(seq.view(len(seq), 1, -1), self.hidden)
        outdat_in_last_timestep=lstm_out[-1, :, :]
        outdat = self.hidden2out(outdat_in_last_timestep)
        return outdat
