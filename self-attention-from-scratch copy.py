import torch
from torch import nn
import torch.nn.functional as F

# Basic Attention
# Vector of form (batch, sequence, dimension)
#    |      d       |    |      d       |
#   s|      d       |   s|      d       |
#    |      d       |,   |      d       |, ...
#                      b
x = torch.rand(2, 3, 3)

# torch.bmm is batched Matrix Multiplication
# Get weight matrix by x*x`
raw_weights = torch.bmm(x, x.transpose(1,2))

# Turn raw weights into positive values that sum to 1 (dim=2) refers to row-wise softmax
weights = F.softmax(raw_weights, dim=2)

# Final Output of form (batch, y sequence, dimension) where rows are weighted sums of weights and x
y = torch.bmm(weights, x)

class SelfAttention(nn.Module):
    def  __init__(self, embed_size, heads=8):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed Size needs to be divisible by heads"

        self.toqueries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.tokeys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.tovalues = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fully_connected_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, key, value, mask=None):
        # N is the batch size b or number of training examples
        N = query.shape[0]
        # These lengths are the sequence lengths s
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = self.toqueries(query)
        key = self.tokeys(key)
        value = self.tovalues(value)
        
        # Turn from (batch, sequence, embed) to (batch, sequence, heads, head dimension)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        values = value.reshape(N, value_len, self.heads, self.head_dim)

        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Attention(Q,K,V) = softmax((QKt)/sqrt(d))*V
        # Softmax normalized over Key Length
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim)

        # Reshape to (N, query_len, embed_size) or (batch, sequence, embed)
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fully_connected_out(out)

        return out

sa1 = SelfAttention(256,8)
print(sa1.toqueries.shape)