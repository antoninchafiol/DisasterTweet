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
        out, (h_out, _) = self.lstm1(x, (h_0, c_0))

        # Concatenating bidirectionnal layers method --> Work but dsoesn't look better than 
        # common unidirectional one and not 100% sure if it's the rigfht way
        o1 = h_out[-1]
        o2 = h_out[-2]
        o3 = torch.cat((o1,o2),1).squeeze(0)
        print(o3.shape)

        # View method. and/or using same as before: Kinda Work (Never goes higher than 60%) 
        h_out = h_out.view(self.num_layers , batch_size, self.hidden_dim * self.bidirectional_val)
        h_out = h_out[-1]
        h_out = h_out.squeeze(0)


        out = self.dropout(o3)
        out = self.fc1(out)
        out = self.output(out)
        return out.squeeze(1) 



# class MultiLSTM(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout, device ):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.device = device
#         self.bidirectional_val = 2 if bidirectional == True else 1 

#         self.lstm1 = torch.nn.LSTM(input_dim, 
#                                    hidden_dim, 
#                                    num_layers = num_layers,
#                                    bidirectional = bidirectional,
#                                    batch_first = True
#                                    )
#         self.fc1 = torch.nn.Linear(hidden_dim * num_layers * self.bidirectional_val, (hidden_dim * num_layers * self.bidirectional_val)/2) 
#         self.dropout1 = torch.nn.Dropout(dropout)
#         self.lstm2 = torch.nn.LSTM(input_dim, 
#                                    hidden_dim, 
#                                    num_layers = num_layers,
#                                    bidirectional = bidirectional,
#                                    batch_first = True
#                                    )
#         self.fc2 = 
#         self.output= torch.nn.Sigmoid()
#     def forward(self, x):
#         h_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
#                                    x.size(0),  self.hidden_dim).to(self.device))
#         c_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
#                                    x.size(0),  self.hidden_dim).to(self.device))

#         # Propagate input through LSTM
#         out, (h_out, _) = self.lstm1(x, (h_0, c_0))
#         out = out[:, -1, :]
#         h_out = h_out.view(-1, self.hidden_dim * self.num_layers * self.bidirectional_val)
#         out = self.fc1(h_out)
#         out = self.output(out)
#         return out.squeeze(1)