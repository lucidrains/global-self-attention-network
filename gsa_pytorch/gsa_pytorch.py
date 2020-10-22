import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from inspect import isfunction

# helpers

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None

def calc_reindexing_tensor(l, L, device):
    """
    Appendix B - (5)
    """
    x = torch.arange(l, device = device)[:, None, None]
    i = torch.arange(l, device = device)[None, :, None]
    r = torch.arange(-(L - 1), L, device = device)[None, None, :]
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
            num_rel_shifts = 2 * rel_pos_length - 1
            self.norm = nn.BatchNorm2d(dim_key)
            self.rel_rows = nn.Parameter(torch.randn(num_rel_shifts, dim_key))
            self.rel_columns = nn.Parameter(torch.randn(num_rel_shifts, dim_key))

    def forward(self, img):
        b, c, x, y, h, c_out, L, device = *img.shape, self.heads, self.dim_out, self.rel_pos_length, img.device

        qkv = self.to_qkv(img).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c (x y)', h = h), qkv)

        k = k.softmax(dim = -1)
        context = einsum('ndm,nem->nde', k, v)

        content_q = q if not self.norm_queries else q.softmax(dim=-2)

        content_out = einsum('nde,ndm->nem', context, content_q)
        content_out = rearrange(content_out, 'n d (x y) -> n d x y', x = x, y = y)

        # this largely follows the mathematical implementation details
        # spelled out in appendix B (6) - (8)
        if exists(self.rel_pos_length):
            q, v = map(lambda t: rearrange(t, 'n c (x y) -> n c x y', x = x, y = y), (q, v))

            Ix = calc_reindexing_tensor(x, L, device)
            Px = einsum('xir,rd->xid', Ix, self.rel_rows)
            Sx = einsum('ndxy,xid->nixy', q, Px)
            Yh = einsum('nixy,neiy->nexy', Sx, v)

            Yh = self.norm(Yh)

            Iy = Ix if x == y else None
            Iy = default(Iy, lambda: calc_reindexing_tensor(y, L, device))

            Py = einsum('yir,rd->yid', Iy, self.rel_columns)
            Sy = einsum('ndxy,yid->nixy', q, Py)
            rel_pos_out = einsum('nixy,nexi->nexy', Sy, Yh)

            content_out = content_out + rel_pos_out.contiguous()

        content_out = rearrange(content_out, '(b h) c x y -> b (h c) x y', h = h)
        return self.to_out(content_out)
