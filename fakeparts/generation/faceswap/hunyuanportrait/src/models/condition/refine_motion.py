
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
from collections import OrderedDict


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, intensity_embedding_dim: Optional[int] = None):
        super().__init__()

        if intensity_embedding_dim is None:
            intensity_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(intensity_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, intensity_embedding: torch.Tensor
    ):
        emb = self.linear(self.silu(intensity_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, intensity_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, intensity_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, intensity_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, intensity_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        intensity_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents, intensity_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, intensity_embedding)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, intensity_embedding))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim=None,
        input_dim=None,
        intensity_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.intensity_aware_linear = nn.Linear(
            intensity_embedding_dim or width, width, bias=True
        )

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, intensity_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, intensity_embedding: torch.Tensor = None):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.intensity_aware_linear(
            torch.nn.functional.silu(intensity_embedding)
        )
        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, intensity_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents


class IntensityAwareMotionRefiner(nn.Module):
    def __init__(
        self,
        input_dim=512,
        output_dim=1024,
        num_queries=64,
        intensity_embed_dim=768,
        width=768,
        layers=6,
        heads=8,
    ):
        super().__init__()
        self.position_head = nn.Embedding(128, intensity_embed_dim)
        self.position_exp = nn.Embedding(128, intensity_embed_dim)
        self.connector = PerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_queries,
            input_dim=input_dim,
            intensity_embedding_dim=intensity_embed_dim,
            output_dim=output_dim,
        )

    def forward(self, motion_features, head_id, exp_id):
        """
        :param motion_features: [batch_size, temporal, num_tokens, channel]
        """        
        dtype = motion_features.dtype

        is_dim2 = len(motion_features.shape) == 2
        if is_dim2:
            motion_features = motion_features.unsqueeze(0).unsqueeze(2)

        B, T = motion_features.shape[:2]  
        motion_features = rearrange(motion_features, "b t n c -> (b t) n c")

        intensity_head = self.position_head(head_id).to(dtype=dtype)
        intensity_exp = self.position_exp(exp_id).to(dtype=dtype)
        intensity_feature = intensity_head + intensity_exp
        intensity_feature = (
            intensity_feature.unsqueeze(dim=0)
            if intensity_feature.ndim == 2
            else intensity_feature
        )
        intensity_feature = intensity_feature.expand(len(motion_features), -1, -1)

        encoder_hidden_states = self.connector(
            motion_features, intensity_feature
        )

        y = rearrange(encoder_hidden_states, "(b t) n c -> b t n c", b=B, t=T)
        if is_dim2:
            y = y.squeeze(0)
        return y
