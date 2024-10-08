import torch
from torch import Tensor, nn
import torch.nn.functional as f


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """Scaled dot product for attention."""
    # bmm Performs a batch matrix-matrix product of matrices stored in input and mat2.
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    """Attention head, part of multi head attention."""

    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        """Init function."""
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Forward pass method."""
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    """Multi head attention, part of transformer network."""

    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        """Init function."""
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Forward pass method."""
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )
