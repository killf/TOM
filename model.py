from torch.nn import functional as F
from torch import nn
import torch
import math


def RoPE(x: torch.Tensor, offset: int = 0):
    """
    旋转位置编码
    x: (batch_size, seq_len, embed_dim)
    """
    _, seq_len, embed_dim  = x.size()
    
    freqs = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2, dtype=torch.float32, device=x.device, requires_grad=False) / embed_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=x.device, requires_grad=False) + offset
    
    freqs = torch.outer(t, freqs)
    freqs_cos, freqs_sin = torch.cos(freqs), torch.sin(freqs)
    
    x_r, x_i = x.reshape(x.shape[:-1] + (-1, 2)).unbind(-1)
    out_r = x_r * freqs_cos - x_i * freqs_sin
    out_i = x_r * freqs_sin + x_i * freqs_cos
    
    return torch.stack([out_r, out_i], dim=-1).flatten(-2)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        
        self.q = nn.Linear(embed_dim, head_dim, bias=False)
        self.k = nn.Linear(embed_dim, head_dim, bias=False)
        self.v = nn.Linear(embed_dim, head_dim, bias=False)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        q, k, v = self.q(x), self.k(x), self.v(x)
       
        q, k = RoPE(q), RoPE(k)
        score = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))
                
        mask = torch.tril(torch.ones_like(score)) == 0
        score = score.masked_fill(mask, - torch.inf)

        weight = torch.softmax(score, -1)
        weight = weight.masked_fill(padding_mask, 0)        
        return torch.bmm(weight, v)
        

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head):
        super().__init__()
        
        head_dim = embed_dim // num_head
        assert embed_dim == head_dim * num_head
        
        self.heads = nn.ModuleList(AttentionHead(embed_dim, head_dim) for _ in range(num_head))
        self.fc = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        x = torch.cat([head(x, padding_mask) for head in self.heads], dim=-1)
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
                
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout: float, norm_eps: float):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_head)
        self.feed_forward = MLP(embed_dim, embed_dim // 2, dropout)
        
        self.attention_norm = nn.RMSNorm(embed_dim, norm_eps)
        self.feed_forward_norm = nn.RMSNorm(embed_dim, norm_eps)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        h = x + self.attention(self.attention_norm(x), padding_mask)
        return h + self.feed_forward(self.feed_forward_norm(h))


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_head: int, n_layers: int, dropout: float=0., norm_eps: float=1e-4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(DecoderLayer(embed_dim, num_head, dropout, norm_eps))
        
        self.norm = nn.RMSNorm(embed_dim, norm_eps)
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.embedding.weight = self.output.weight 

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len) int32
        y: (batch_size, seq_len) int32
        """
        padding_mask = get_padding_mask(x == 0)
        x = self.dropout(self.embedding(x))
        
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.norm(x)
        
        if y is not None:
            p = self.output(x)
            p, y = p.view(-1, p.size(-1)), y.flatten()
            loss = F.cross_entropy(p, y)
            return p, loss
        else:
            return self.output(x[:, -1, :])
        

def get_padding_mask(mask):
    """
    mask: (batch_size, seq_len) bool
    """
    _, seq_len = mask.shape
    p1 = mask.unsqueeze(-1).repeat(1, 1, seq_len)
    p2 = mask.unsqueeze(-2).repeat(1, seq_len, 1)
    return torch.logical_or(p1, p2)
    

if __name__ == "__main__":
    # x = torch.randint(0, 3000, (1, 10))
    # y = torch.randint(0, 3000, (1, 10))
    # net = Transformer(3000, 64, 8, 2)
    # p, loss = net(x, y)
    # print(p.shape, loss)
    
    x = torch.rand((2, 10, 64))
    m = torch.tensor([[False] * 5 + [True] * 5, [False] * 7 + [True] * 3], dtype=torch.bool)
    
    net = AttentionHead(64, 8)
    y1 = net(x, get_padding_mask(m))
    y2 = net(x[:, :8, :], get_padding_mask(m[:, :8]))
    print(y1.shape)
    print(y1[0, 7])
    print(y1[0, 7] == y2[0, 7])
    

        
    