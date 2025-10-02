"""
Building Blocks for GABI Models

Reusable neural network components including:
- Transformer blocks with classic and Galerkin attention
- Attention mechanisms
"""

import torch
from torch import nn
from einops import rearrange
from torch_geometric.nn import MLP
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, attn_type, dim, heads, dim_head, mlp_dim, qkv_bias=False, drop=0., attn_drop=0., attn_layer_norm=True, mlp_layer_norm=True):
        super().__init__()
        
        assert attn_type in {'classic', 'galerkin'}, 'Attention type must be either classic, galerkin or fourier'
            
        if attn_type == 'classic':
            self.attn = ClassicAttention(dim, heads, dim_head, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, layer_norm=attn_layer_norm)
        elif attn_type == 'galerkin':
            self.attn = GalerkinAttention(dim, heads, dim_head, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, layer_norms=attn_layer_norm)

        self.mlp_layer_norm  = nn.LayerNorm(dim, eps=1e-6) if mlp_layer_norm else nn.Identity()
        self.mlp = MLP(in_channels=dim, hidden_channels=mlp_dim, out_channels=dim, num_layers=2, dropout=drop, act='gelu', norm=None)
        
    def forward(self,  x, return_internal=False):               
        if self.attn is not None:
            if return_internal:
                z, attn_, q, k, v, z_pre_proj = self.attn(x, return_qkvz=True)
            else:
                z = self.attn(x)
                
            z_add_x = z  + x
            
        z_ff = self.mlp(self.mlp_layer_norm(z_add_x))
        x = z_ff + z_add_x
        
        if return_internal:
            return x, attn_, q, k, v, z_pre_proj, z, z_ff, z_add_x
        else:
            return x
        

class ClassicAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, qkv_bias=False, attn_drop=0., proj_drop=0., layer_norm=True):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.norm = nn.LayerNorm(dim, eps=1e-6) if layer_norm else nn.Identity()

        self.heads = heads
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = attn_drop
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(proj_drop)
        ) if project_out else nn.Identity()

    def forward(self, x, return_qkvz=False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)

        # dots = einsum('b h i d, b h j d -> b h i j', q, k)
        
        # attn = self.attend(dots)
        # attn = self.attn_drop(attn)

        # out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_qkvz:
            return self.to_out(out), q, k, v, out
        else:
            return self.to_out(out)

class GalerkinAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, qkv_bias=False, attn_drop=0., proj_drop=0., layer_norms=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        if layer_norms:
            self.norm1 = nn.LayerNorm(dim_head, eps=1e-6)
            self.norm2 = nn.LayerNorm(dim_head, eps=1e-6)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(proj_drop)
        ) if project_out else nn.Identity()

    def forward(self, x, return_qkvz=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        k = self.norm1(k)
        v = self.norm1(v)
        
        # scale q by dividing by n
        q = q/n
        
        ktv = torch.matmul(k.permute(0,1,3,2), v)
        out = torch.matmul(q, ktv)
            
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_qkvz:
            return self.to_out(out), ktv, q, k, v, out
        else:
            return self.to_out(out)