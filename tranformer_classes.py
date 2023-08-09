
import torch

class Embedder(torch.nn.Module):
    '''
    The Embedding layer in an allocated class
    '''
    def __init__(self, params, d_model) -> None:
        '''
        Initialize and freeze the embedding layer

        Parameters
        ----------
        params --> ['vocab_size']: dict --> int
            All params related to model --> Size of the vocabulary
        d_model: int
            Dimension of the model
        '''
        super().__init__()
        self.embed = torch.nn.Embedding(params['vocab_size'], d_model)
        self.embed.weight.data.copy_(d_model)
        self.embed.weight.requires_grad = False
    def forward(self, x):
        return self.embed(x)
    
class PositionalEncoder(torch.nn.Module):
    def __init__(self, max_seq_length, d_model):
        super().__init__()
        self.d_model = d_model
        self.pe = torch.zeros(max_seq_length, d_model)
        
    def forwward(self, x):
        return x