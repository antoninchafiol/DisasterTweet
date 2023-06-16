import torch
   
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