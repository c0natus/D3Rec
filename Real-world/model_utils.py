import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class PosEmb(nn.Module):
    def __init__(self, pos_dim):
        super(PosEmb, self).__init__()
        """
        Sinusoidal timestep positional encoding.
        """
        self.pos_dim = pos_dim
        self.time_mlp = nn.Linear(pos_dim, pos_dim)

    def forward(self, t, max_period=10000):
        t = t.unsqueeze(-1).type(torch.float)
        half_dim = self.pos_dim // 2
        w_k = 1.0 / (
            max_period
            ** (torch.arange(0, half_dim, 1, device=t.device).float() / (half_dim-1))
        )

        half_emb = t.repeat(1, half_dim)
        pos_sin = torch.sin(half_emb * w_k)
        pos_cos = torch.cos(half_emb * w_k)
        pos_enc = torch.cat([pos_sin, pos_cos], dim=-1)

        emb = self.time_mlp(pos_enc)
        return emb


class Encoder(nn.Module):
    def __init__(self, n_item, dims, dim_step, proj_layer, cond_dim):
        super(Encoder, self).__init__()

        ind_in_dims = [n_item + dim_step] + [dim // 2 for dim in dims[:-1]]
        ind_out_dims = [dim // 2 for dim in dims]
        self.half_dims = ind_out_dims
        self.cond_dim = cond_dim

        cate_in_dims = [n_item + dim_step] + [dim // 2 for dim in dims[:-1]]
        cate_out_dims = [dim // 2 - dim // 4 for dim in dims]

        self.encoder_ind = nn.ModuleList([])
        self.encoder_cate = nn.ModuleList([])
        self.proj_layer = proj_layer

        for d_in, d_out in zip(ind_in_dims, ind_out_dims):
            self.encoder_ind.append(
                nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.Tanh()
                ))

        for d_in, d_out in zip(cate_in_dims, cate_out_dims):
            self.encoder_cate.append(
                nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.Tanh()
                ))

    def forward(self, x_t, emb_cate):
        latent_z = x_t
        latent_h= x_t

        for idx in range(len(self.encoder_ind)):
            latent_z = self.encoder_ind[idx](latent_z)
            latent_h = self.encoder_cate[idx](latent_h)

            if latent_h.size(-1) != self.cond_dim:
                projected_emb = self.proj_layer[idx](emb_cate)
            else:
                projected_emb = emb_cate

            latent = torch.cat([latent_z, latent_h, projected_emb], dim=-1)

            latent_z = latent[:, :self.half_dims[idx]]
            latent_h = latent[:, self.half_dims[idx]:]

        return torch.cat([latent_z, latent_h], dim=-1)


class Decoder(nn.Module):
    def __init__(self,  n_item, dims, proj_layer, cond_dim):
        super(Decoder, self).__init__()

        in_dims = [dim for dim in dims[::-1]]
        out_dims = [dim - dim // 4 for dim in in_dims[1:]] + [n_item]
        self.cond_dim = cond_dim

        self.proj_layer = proj_layer
        self.decode_layer = nn.ModuleList([])

        for d_in, d_out in zip(in_dims, out_dims):
            self.decode_layer.append(
                nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.Tanh(),
                ))

        # Remove last activation
        self.decode_layer[-1] = self.decode_layer[-1][:-1]

    def forward(self, latent, emb_cate):
        for idx in range(len(self.decode_layer) - 1):
            latent = self.decode_layer[idx](latent)
            latent = torch.cat([latent, self.proj_layer[idx](emb_cate)], dim=-1)

        out = self.decode_layer[-1](latent)
        return out


class MyMLP(nn.Module):
    def __init__(self, n_item, n_cate, dims, dim_step, dropout):
        super(MyMLP, self).__init__()

        self.dims = dims
        self.latent_half = dims[-1] // 2
        self.drop = nn.Dropout(dropout)

        cond_dim = dims[-1] // 4
        cond_dims = [dim // 2 - dim // 4 for dim in dims[:-1]]
        self.proj_layer = nn.ModuleList([])
        for proj_dim in cond_dims:
            self.proj_layer.append(nn.Linear(cond_dim, proj_dim))

        self.encoder = Encoder(n_item, dims, dim_step, self.proj_layer, cond_dim)

        self.cate_embedding = nn.Embedding(n_cate, dims[-1] // 4)
        self.cate_zero_tensor = nn.Parameter(torch.zeros(n_cate, dims[-1] * 3 // 4), requires_grad=False)

        self.decoder = Decoder(n_item, dims, self.proj_layer[::-1], cond_dim)

    def forward(self, x_t, probs, probs_mask):
        x_t = self.drop(x_t)

        emb_pref = torch.einsum('bn, np -> bp', probs, self.cate_embedding.weight)
        emb_pref[probs_mask == 1, :] = 0

        if self.training:
            latent = self.encoder(x_t, emb_pref)
            x_hat_recon = self.decoder(latent, emb_pref)
            ortho_loss = self.compute_orthogonal(
                latent[:, :self.latent_half],
                latent[:, self.latent_half:])

            emb_div = latent[:, self.latent_half:]
            latent_prob = torch.cat([torch.zeros_like(emb_div), emb_div], dim=-1)
            latent_cate = torch.cat([self.cate_zero_tensor, self.cate_embedding.weight], dim=-1)
            x_hat_prob = self.decoder(latent_prob, emb_pref)
            x_hat_cate = self.decoder(latent_cate, self.cate_embedding.weight)

            return x_hat_recon, x_hat_prob, x_hat_cate, ortho_loss
        else:
            latent = self.encoder(x_t, emb_pref)
            x_hat_recon = self.decoder(latent, emb_pref)
            return x_hat_recon

    def compute_orthogonal(self, z, w):
        z = F.normalize(z, p=2, dim=-1)
        w = F.normalize(w, p=2, dim=-1)
        loss = torch.abs(torch.einsum('ix, ix -> i', z, w))
        return loss.sum()


class AutoRec(nn.Module):
    def __init__(self, n_item, n_cate, dims, dim_step, dropout, act):
        super(AutoRec, self).__init__()

        self.dims = dims
        self.latent_half = dims[-1] // 2
        self.drop = nn.Dropout(dropout)

        if act == 'tanh': act_f = nn.Tanh()
        elif act == 'elu': act_f = nn.ELU()
        elif act == 'silu': act_f = nn.SiLU()
        elif act == 'gelu': act_f = nn.GELU()
        elif act == 'relu': act_f = nn.ReLU()

        cond_dim = dims[-1] // 4

        cond_dims = [dim // 2 - dim // 4 for dim in dims[:-1]]
        self.proj_emb = nn.ModuleList([])
        for liner_out in cond_dims:
            self.proj_emb.append(nn.Linear(cond_dim, liner_out))

        self.encoder = Encoder(n_item, dims, dim_step, self.proj_emb, cond_dim, act_f)

        self.cate_embedding = nn.Embedding(n_cate, dims[-1] // 4)
        self.cate_zero_tensor = nn.Parameter(torch.zeros(n_cate, dims[-1] * 3 // 4), requires_grad=False)

        self.decoder = Decoder(n_item, dims, self.proj_emb[::-1], cond_dim, act_f)

    def forward(self, x_t, probs, probs_mask):
        x_t = self.drop(x_t)

        emb_pref = torch.einsum('bn, np -> bp', probs, self.cate_embedding.weight)
        emb_pref[probs_mask == 1, :] = 0

        if self.training:
            latent = self.encoder(x_t, emb_pref)
            x_hat_recon = self.decoder(latent, emb_pref)
            ortho_loss = self.compute_orthogonal(
                latent[:, :self.latent_half],
                latent[:, self.latent_half:])

            emb_div = latent[:, self.latent_half:]
            latent_prob = torch.cat([torch.zeros_like(emb_div), emb_div], dim=-1)
            latent_cate = torch.cat([self.cate_zero_tensor, self.cate_embedding.weight], dim=-1)
            x_hat_prob = self.decoder(latent_prob, emb_pref)
            x_hat_cate = self.decoder(latent_cate, self.cate_embedding.weight)

            return x_hat_recon, x_hat_prob, x_hat_cate, ortho_loss
        else:
            latent = self.encoder(x_t, emb_pref)
            x_hat_recon = self.decoder(latent, emb_pref)
            return x_hat_recon

    def compute_orthogonal(self, z, w):
        z = F.normalize(z, p=2, dim=-1)
        w = F.normalize(w, p=2, dim=-1)
        loss = torch.abs(torch.einsum('ix, ix -> i', z, w))
        return loss.sum()