import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# helpers

def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None

def calc_reindexing_tensor(l, L, device):
    x = torch.arange(l, device = device)[:, None, None]
    i = torch.arange(l, device = device)[None, :, None]
    r = torch.arange(L, device = device)[None, None, :]
    mask = ((i - x) == r) & ((i - x).abs() <= L)
    return mask.float()

# classes

class GSA(nn.Module):
    def __init__(self, dim, *, rel_pos_length = None, dim_out = None, heads = 8, dim_key = 64, norm_queries = False):
        super().__init__()
        dim_out = default(dim_out, dim)
        dim_hidden = dim_key * heads

        self.heads = heads
        self.dim_out = dim_out
        self.rel_pos_length = rel_pos_length
        self.norm_queries = norm_queries

        self.to_qkv = nn.Conv2d(dim, dim_hidden * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim_hidden, dim_out, 1)

        self.rel_pos_length = rel_pos_length
        if exists(rel_pos_length):
            self.norm = nn.BatchNorm2d(dim_key)
            self.rel_rows = nn.Parameter(torch.randn(rel_pos_length, dim_key))
            self.rel_columns = nn.Parameter(torch.randn(rel_pos_length, dim_key))

    def forward(self, img):
        b, c, x, y, h, c_out, L, device = *img.shape, self.heads, self.dim_out, self.rel_pos_length, img.device

        qkv = self.to_qkv(img).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c x y', h = h), qkv)

        k = k.softmax(dim = -1)
        context = einsum('ndxy,nexy->nde', k, v)

        content_q = q if not self.norm_queries else q.softmax(dim=-2)

        content_out = einsum('nde,ndxy->nexy', context, content_q)

        if exists(self.rel_pos_length):
            Ix = calc_reindexing_tensor(x, L, device)
            Px = einsum('xir,rd->xid', Ix, self.rel_rows)
            Sx = einsum('ndxy,xid->nixy', q, Px)
            Yh = einsum('nixy,neiy->nexy', Sx, v)
            del Ix

            Yh = self.norm(Yh)

            Iy = calc_reindexing_tensor(y, L, device)
            Py = einsum('xir,rd->xid', Iy, self.rel_columns)
            Sy = einsum('ndxy,xid->nixy', q, Py)
            rel_pos_out = einsum('nixy,neiy->nexy', Sy, Yh)
            del Iy

            content_out = content_out + rel_pos_out

        content_out = rearrange(content_out, '(b h) c x y -> b (h c) x y', b = b, h = h)
        return self.to_out(content_out)
