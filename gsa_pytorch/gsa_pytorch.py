import torch
from torch import nn, einsum
from einops import rearrange

# helpers

def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None

# classes

class GSA(nn.Module):
    def __init__(self, dim, *, rel_pos_length = None, dim_out = None, heads = 8, dim_key = 64):
        super().__init__()
        dim_out = default(dim_out, dim)
        dim_hidden = dim_key * heads

        self.heads = heads
        self.dim_out = dim_out
        self.rel_pos_length = rel_pos_length

        self.to_qkv = nn.Conv2d(dim, dim_hidden * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim_hidden, dim_out, 1)

        if exists(rel_pos_length):
            self.rel_columns = nn.Parameter(torch.randn(rel_pos_length, dim_key))
            self.rel_rows = nn.Parameter(torch.randn(rel_pos_length, dim_key))

    def forward(self, img):
        b, c, x, y, h, c_out = *img.shape, self.heads, self.dim_out

        qkv = self.to_qkv(img).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = h), qkv)

        k = k.softmax(dim = -1)
        context = einsum('bhdn,bhen->bhde', k, v)

        content_out = einsum('bhde,bhdn->bhen', context, q)
        content_out = content_out.reshape(b, -1, x, y)
        content_out = self.to_out(content_out)

        # todo: compute relative position attentions and sum to content_out

        return content_out
