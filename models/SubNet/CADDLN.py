"""CADDLN network definition."""

import torch
from models.Layers.CADDLN_Layers import *


class CADDLN(nn.Module):

    def __init__(
        self, dict_dim: int, grid_num: int, d_model: int = 64, nhead: int = 2, num_layers: int = 2
    ):
        super().__init__()
        self.dict_dim = dict_dim
        self.grid_num = grid_num
        self.d_model = d_model

        self.q_proj = nn.Sequential(
            nn.Linear(2 * dict_dim, dict_dim * d_model), nn.LayerNorm(dict_dim * d_model), nn.GELU()
        )

        self.kv_proj = nn.Sequential(
            nn.Linear(2 * grid_num, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=max(dict_dim, grid_num))

        self.cross_layers = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(d_model, nhead, dim_feedforward=d_model * 2)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Sequential(nn.Linear(d_model, 2 * grid_num), nn.Tanh())

    def forward(self, y_vec: torch.Tensor, oc_dict: torch.Tensor) -> torch.Tensor:

        batch_size = y_vec.shape[0]

        y_real = y_vec.real
        y_imag = y_vec.imag
        y_cat = torch.cat([y_real, y_imag], dim=-1)

        q_seq = self.q_proj(y_cat).reshape(batch_size, self.dict_dim, self.d_model)

        o_real = oc_dict.real
        o_imag = oc_dict.imag
        o_cat = torch.cat([o_real, o_imag], dim=-1)

        kv_seq = self.kv_proj(o_cat)

        q_seq = self.pos_encoder(q_seq)
        kv_seq = self.pos_encoder(kv_seq)

        x = q_seq
        for layer in self.cross_layers:
            x = layer(x, kv_seq)

        x_out = self.output_proj(x)
        x_out_real = x_out[:, :, : self.grid_num]
        x_out_imag = x_out[:, :, self.grid_num :]
        delta = torch.complex(x_out_real, x_out_imag)

        delta_norms = torch.norm(delta, p=2, dim=1, keepdim=True)
        delta_normalized = delta / delta_norms

        final_dict = oc_dict + delta_normalized
        final_dict_norms = torch.norm(final_dict, p=2, dim=1, keepdim=True)
        final_dict = final_dict / final_dict_norms

        return final_dict
