
import torch
import torch.nn.functional as F

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
    '''
    Class for computing and forward pass the positional encoding 
    '''
    def __init__(self, max_seq_length, d_model):
        '''
        Construct the PE matrix and register it as a buffer

        Parameters
        ----------
        max_seq_length: int
            Maximum length of the sentence
            
        d_model: int
            Dimension of the model
        '''
        super().__init__()
        self.d_model = d_model 
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i]   = torch.sin(pos / torch.pow(torch.tensor(10000), 
                                                         torch.tensor( (2*i)/d_model )))
                pe[pos, i+1] = torch.cos(pos / torch.pow(torch.tensor(10000), 
                                                         torch.tensor( (2*(i+1))/d_model )))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class MultiHeadAttention(torch.nn.Module):
    '''
    Create and compute Multi-Head Attention with "embedded" Scaled Dot-Product Attention 
    '''
    def __init__(self, heads, d_model, p_drop=0.1):
        '''
        Constructing the differents layers and variables 

        Parameters
        ----------
        heads: int
            Number of heads 
        d_model: int
            Dimension of the model/embedding
        p_drop: float
            Dropout rate 
        '''
        super().__init__()
        self.d_model = d_model
        self.h = heads
        self.d_k = d_model // heads
        self.p_drop = p_drop

        self.q = torch.nn.Linear(d_model, d_model)
        self.k = torch.nn.Linear(d_model, d_model)
        self.v = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(p_drop)
        self.output  = torch.nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q,k,v, d_k, mask=None, dropout=None):
        '''
        Compute the scaled dot-product Attention and apply mask and dropout if given.

        Parameters
        ----------
        q: tensor
            Queries matrix
        k: tensor
            Keys matrix
        v: tensor
            Values Matrix
        d_k: int
            Final dimension 
        mask: tensor
            Mask to apply to computed matmul & sqrt
        dropout: torch.nn.Dropout
            Dropout to apply to Attention weights
        ''' 
        # We tranpose K for matmul 
        # Shape wanted for K.T: (batch_size, num_heads, embed_dim, seq_len)
        scores = torch.matmul(q,k.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k))

        if mask is not None:
            mask.unsqueeze(1)
            scores = scores.masked_fill(mask==0, -1e9)
        
        attention_weights = F.softmax(scores, dims=-1)

        if dropout is not None:
            attention_weights = dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        
        return output
    
    def forward(self,q,k,v, mask=None):
        '''
        Apply the forward computation by splitting matrices per heads, transposing to right dimensions, 
        apply Attention and concatenating the result.

        Parameters
        ----------
        q: tensor
            Queries matrix
        k: tensor
            Keys matrix
        v: tensor
            Values Matrix
        mask: tensor
            Mask to apply to computed matmul & sqrt
        '''
        batch_size = q.size(0) # Shape of matrices : (batch_size, seq_len, embed_dim)
        # Using .view() to reshape the tensors in shape (batch_size, seq_len, num_heads, embed_dim)
        V = self.v(v).view(batch_size, -1, self.h, self.d_k)
        K = self.k(k).view(batch_size, -1, self.h, self.d_k)
        Q = self.q(q).view(batch_size, -1, self.h, self.d_k)

        # Transposing matrices to have the right matrices shape for future usage 
        # Shape wanted : (batch_size, num_heads, seq_len, embed_dim)
        K = K.transpose(1,2)
        Q = Q.transpose(1,2)
        V = V.transpose(1,2)

        attention_score = self.scaled_dot_product_attention(Q,K,V, self.d_k, mask=None, dropout=self.dropout) 

        concat = attention_score.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        output = self.output(concat)
        return output
    
class FeedForward(torch.nn.Module):
    '''
    2 Linear layers separated by ReLu activation and dropout.
    Serve as identifying pattern and deepens the network. 
    '''
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        '''
        Construct the linear layers and the dropout layers with given parameters.

        Parameters
        ----------
        d_model: int
            Dimension of the model/embedding
        d_ff: int
            Dimension of the second linear layer's input
        dropout: float
            Rate of dropout to apply
        '''
        super().__init__() 
        self.fc1     = torch.nn.Linear(d_model, d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2     = torch.nn.Linear(d_ff, d_model)
    def forward(self, x):
        '''
        Compute the linear layers and apply ReLu/dropout
        '''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class LayerNorm(torch.nn.Module):
    '''
    Create and compute the layer normalization
    '''
    def __init__(self, d_model, epsilon=1e-6):
        '''
        Initialize the learnable parameters and store epsilon's value
        
        Parameters
        ----------
        d_model: int
            Dimension of the model/embedding 
        epsilon: float
            Value of epsilon 
        '''
        super().__init__()
        self.size = d_model
        self.alpha = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = epsilon
        
    def forward(self, z):
        '''
        Compute the layer normalization as stated '
        '''
        mean = z.mean(-1, keepdim=True)
        std = z.std(-1, keepdim=True)
        mu = z-mean
        sigma = std + self.eps
        norm = (z-mu / sigma) * self.alpha + self.beta
        return norm



