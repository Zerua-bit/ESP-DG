from torch import nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import math


class ChannelSelfAttention(nn.Module):
    def __init__(self, dim, attn_drop=0., proj_drop=0., num_heads=8):
        super().__init__()

        self.scale = (dim) ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.num_heads = num_heads

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat):
        B, N, C = feat.shape

        q = self.q(feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        k = self.k(feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).reshape(B, -1, N).permute(0, 2, 1)
        x = self.proj_drop(self.proj(x))

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim2, bias=qkv_bias)
        self.k = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.v = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape

        q = self.q(x).reshape(B1, N1, self.num_heads, C2 // self.num_heads).permute(0, 2, 3, 1)
        k = self.k(y).reshape(B2, N2, self.num_heads, C2 // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(y).reshape(B2, N2, self.num_heads, C2 // self.num_heads).permute(0, 2, 3, 1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v).reshape(B1, -1, N1).permute(0, 2, 1)
        y = self.proj(y)
        y = self.proj_drop(y)

        return y

class Block_feed(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim1)
        self.norm3 = norm_layer(dim2)
        self.norm4 = norm_layer(dim2)
        self.selfattn = ChannelSelfAttention(dim1)
        self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H1, W1):
        x = x + self.drop_path(self.selfattn(self.norm1(x)))
        y = y + self.drop_path(self.attn(self.norm2(x), self.norm3(y)))
        y = y + self.mlp(self.norm4(y), H1, W1)
        cat = torch.cat([x, y], dim=-1)
        return cat, y

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.attn_c4_c1 = Block_feed(dim1=up_out, dim2=x_out, num_heads=8, mlp_ratio=4,
                                     drop_path=0.1)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        b4, ch4, h4, w4 = up_p.shape
        b1, ch1, h1, w1 = x_p.shape
        c4 = up_p.flatten(2).transpose(1, 2)
        c1 = x_p.flatten(2).transpose(1, 2)
        cat, y = self.attn_c4_c1(c4, c1, h1, w1)
        y = y.reshape(b1, h1, w1, ch1).permute(0, 3, 1, 2)
        ch = ch1 + ch4
        cat_p = cat.reshape(b1, h1, w1, ch).permute(0, 3, 1, 2)
        return self.bn(F.relu(cat_p)), y


