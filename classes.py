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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.bidirectional_val = 2 if bidirectional == True else 1 

        self.lstm1 = torch.nn.LSTM(input_dim, 
                                   hidden_dim, 
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   batch_first = True
                                   )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_dim * self.bidirectional_val, output_dim) 
        self.output= torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   batch_size,  self.hidden_dim).to(self.device))
        c_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   batch_size,  self.hidden_dim).to(self.device))

        # Propagate input through LSTM
        _, (h_out, _) = self.lstm1(x, (h_0, c_0))

        # Concatenating bidirectionnal layers method --> Work but dsoesn't look better than 
        # common unidirectional one and not 100% sure if it's the rigfht way
        h_out = torch.cat((h_out[-1],h_out[-2]), dim=-1).squeeze(0)
        out = self.dropout(h_out)
        out = self.fc1(out)
        out = self.output(out)
        print(out.shape)
        return out.squeeze(1) 



class MultiLSTM(torch.nn.Module):
    def __init__(self, input_dim, input_dim2, hidden_dim, hidden_dim2, output_dim, num_layers, bidirectional, dropout, device ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.num_layers = num_layers
        self.device = device
        self.bidirectional_val = 2 if bidirectional == True else 1 

        self.lstm1 = torch.nn.LSTM(input_dim, 
                                   hidden_dim, 
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   batch_first = True
                                   )
        self.fc1 = torch.nn.Linear(hidden_dim * self.bidirectional_val, input_dim2) 
        self.dropout1 = torch.nn.Dropout(dropout)
        self.lstm2 = torch.nn.LSTM(input_dim2, 
                                   hidden_dim2, 
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   batch_first = True
                                   )
        self.fc2 = torch.nn.Linear(hidden_dim2 * self.bidirectional_val, output_dim)
        self.output= torch.nn.Sigmoid()
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   x.size(0),  self.hidden_dim).to(self.device))
        c_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   x.size(0),  self.hidden_dim).to(self.device))

        # Propagate input through LSTM
        _, (h_out, _) = self.lstm1(x, (h_0, c_0))
        h_out = torch.cat((h_out[-1],h_out[-2]), dim=-1).squeeze(0)
        out = self.dropout1(h_out)
        out = self.fc1(out)

        h_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   out.shape[0],  self.hidden_dim2).to(self.device))
        c_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   out.shape[0],  self.hidden_dim2).to(self.device))
        
        out = torch.reshape(out, (out.size(0), 1,  out.size(1)))
        _, (h_out, _) = self.lstm2(out, (h_0, c_0))
        h_out = torch.cat((h_out[-1],h_out[-2]), dim=-1).squeeze(0)
        out = self.fc2(out)
        out = self.output(out)
        out = out.squeeze(1)
        return out.squeeze(1)