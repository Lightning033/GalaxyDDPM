import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t[:, None].float() * emb[None, :]
        emb    = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, emb_dim)

    def forward(self, c):
        return self.embedding(c)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.dropout  = nn.Dropout(dropout)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act      = nn.SiLU()

    def forward(self, x, emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(self.act(emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_ch       = 3,
        base_ch     = 128,
        ch_mult     = (1, 2, 2, 4),
        num_classes = 8,
        emb_dim     = 256,
        dropout     = 0.1,
    ):
        super().__init__()

        channels = [base_ch * m for m in ch_mult]

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.class_emb = nn.Sequential(
            ClassEmbedding(num_classes, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        enc_channels     = []

        in_c = base_ch
        for i, out_c in enumerate(channels):
            use_attn = (i == len(channels) - 1)
            self.down_blocks.append(nn.ModuleList([
                ResBlock(in_c,  out_c, emb_dim, dropout),
                ResBlock(out_c, out_c, emb_dim, dropout),
                AttentionBlock(out_c) if use_attn else nn.Identity(),
            ]))
            enc_channels.append(out_c)
            self.downsamples.append(
                Downsample(out_c) if i < len(channels) - 1 else nn.Identity()
            )
            in_c = out_c

        self.mid_block1 = ResBlock(channels[-1], channels[-1], emb_dim, dropout)
        self.mid_attn   = AttentionBlock(channels[-1])
        self.mid_block2 = ResBlock(channels[-1], channels[-1], emb_dim, dropout)

        self.up_blocks  = nn.ModuleList()
        self.upsamples  = nn.ModuleList()

        rev_channels     = list(reversed(channels))
        rev_enc_channels = list(reversed(enc_channels))

        in_c = rev_channels[0]
        for i in range(len(rev_channels)):
            out_c    = rev_channels[i]
            skip_c   = rev_enc_channels[i]
            use_attn = (i == 0)
            self.up_blocks.append(nn.ModuleList([
                ResBlock(in_c + skip_c, out_c, emb_dim, dropout),
                ResBlock(out_c,         out_c, emb_dim, dropout),
                AttentionBlock(out_c) if use_attn else nn.Identity(),
            ]))
            self.upsamples.append(
                Upsample(out_c) if i < len(rev_channels) - 1 else nn.Identity()
            )
            in_c = out_c

        self.output_norm = nn.GroupNorm(8, base_ch)
        self.output_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t, c):
        emb = self.time_emb(t) + self.class_emb(c)
        x   = self.input_conv(x)

        skips = []
        for (rb1, rb2, attn), ds in zip(self.down_blocks, self.downsamples):
            x = rb1(x, emb)
            x = rb2(x, emb)
            x = attn(x)
            skips.append(x)
            x = ds(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for (rb1, rb2, attn), us, skip in zip(
                self.up_blocks, self.upsamples, reversed(skips)):
            x = torch.cat([x, skip], dim=1)
            x = rb1(x, emb)
            x = rb2(x, emb)
            x = attn(x)
            x = us(x)

        x = self.output_conv(F.silu(self.output_norm(x)))
        return x
