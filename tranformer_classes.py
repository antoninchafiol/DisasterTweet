
import torch

class Embedder(torch.nn.Module):
    def __init__(self, params, d_model) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(params['vocab_size'], d_model)
        self.embed.weight.data.copy_(d_model)
        self.embed.weight.requires_grad = False
    def forward(self, x):
        return self.embed(x)
    