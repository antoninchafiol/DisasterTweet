import torch
import torch.nn
from torch.autograd import Variable
   
class SimpleNet(torch.nn.Module):
    def __init__(self, input_len, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_len, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.output= torch.nn.Sigmoid()
    def forward(self, x):
        fc1 = self.fc1(x)
        fc2 = self.fc2(fc1)
        output = self.output(fc2)
        return output[:, -1]


class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout, device ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.bidirectional_val = 2 if bidirectional == True else 1 
        # From PyTorch's Documentation
        # N = Batch_size
        # L = sequence length
        # H_in = input_size
        # H_cell = hidden_size
        
        self.lstm1 = torch.nn.LSTM(input_dim, 
                                   hidden_dim, 
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   dropout = dropout,
                                   batch_first = True
                                   )
        self.fc1 = torch.nn.Linear(hidden_dim, output_dim) 
        self.output= torch.nn.Sigmoid()
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   x.size(0),  self.hidden_dim).to(self.device))
        c_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   x.size(0),  self.hidden_dim).to(self.device))

        # Propagate input through LSTM
        out, (h_out, _) = self.lstm1(x, (h_0, c_0))
        out = out[:, -1, :]
        print(h_out.shape)
        h_out = h_out.view(-1, self.hidden_dim * self.bidirectional_val)
        print(out.shape)
        print(h_out.shape)
        out = self.fc1(h_out)
        out = self.output(out)
        return out.squeeze(1)