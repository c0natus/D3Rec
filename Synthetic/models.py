import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model_utils import *


class Diffusion():
    def __init__(self, steps, beta_start, beta_end, device, noise_schedule,
            noise_scale, num_for_expectation=10, beta_fixed=True):

        self.steps = steps
        self.beta_start = beta_start # noise_min
        self.beta_end = beta_end # noise_max
        self.device = device
        self.noise_scale = noise_scale
        self.noise_schedule = noise_schedule

        self.beta = torch.tensor(self.get_betas(), dtype=torch.float32).to(device)
        if beta_fixed and self.beta[0] > 1e-5:
            self.beta[0] = 1e-5
        # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
        # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.

        assert len(self.beta.shape) == 1, "beta must be 1-D"
        assert len(self.beta) == self.steps, "num of beta must equal to diffusion steps"
        assert (self.beta > 0).all() and (self.beta <= 1).all(), "beta out of range"

        self.caculate_for_diffusion()


        self.num_for_expectation = num_for_expectation
        self.Lt_history = torch.zeros(steps, num_for_expectation, dtype=torch.float32).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

    def get_betas(self):
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.beta_start
            end = self.noise_scale * self.beta_end
            if self.noise_schedule == "linear": return np.linspace(start, end, self.steps, dtype=np.float64)
            else: return self.linear_variance_beta_schedule(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine": # Improved diffusion
            return self.cosin_beta_schedule(self.steps, max_beta=self.beta_end)
        elif self.noise_schedule == "exp":
            start = self.noise_scale * self.beta_start
            end = self.noise_scale * self.beta_end
            return self.exp_beta_schedule(self.steps, beta_min=start, beta_max=end)
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return self.noise_scale * betas
        elif self.noise_schedule == "sqrt":
            return self.sqrt_beta_schedule(self.steps, max_beta=self.beta_end)
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def linear_variance_beta_schedule(self, steps, variance, max_beta=0.999):
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return np.array(betas)

    def cosin_beta_schedule(self, steps, s=0.008, max_beta=0.999):
        alpha_bar = lambda t: (math.cos((t + s) / (1 + s) * math.pi / 2) ** 2)
        betas = []
        for i in range(steps):
            t1 = i / steps
            t2 = (i + 1) / steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)
    
    def exp_beta_schedule(self, steps, beta_min=0.1, beta_max=10):
        x = torch.linspace(1, 2 * steps + 1, steps)
        betas = 1 - torch.exp(- beta_min / steps - x * 0.5 * (beta_max - beta_min) / (steps * steps))
        return np.array(betas)
        
    def sqrt_beta_schedule(self, steps, max_beta=0.999):
        alpha_bar = lambda t: 1-np.sqrt(t + 0.0001)
        betas = []
        for i in range(steps):
            t1 = i / steps
            t2 = (i + 1) / steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)
    
    def caculate_for_diffusion(self):
        alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(alpha, dim=0).to(self.device)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alpha_bar[:-1]]).to(self.device)

        assert self.alpha_bar_prev.shape == (self.steps,)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        self.posterior_mean_coef1 = torch.sqrt(alpha) * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef2 = torch.sqrt(self.alpha_bar_prev) * self.beta / (1.0 - self.alpha_bar)

        self.posterior_variance = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )

    def sample_steps(self, batch_size, method, uniform_prob=0.001):
        if method == 'importance':
            if not (self.Lt_count == self.num_for_expectation).all():
                return self.sample_steps(batch_size, method='uniform')

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))

            pt_prob = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_prob *= 1 - uniform_prob 
            pt_prob += uniform_prob / len(pt_prob)

            assert pt_prob.sum(-1) - 1. < 1e-5

            steps = torch.multinomial(input=pt_prob, num_samples=batch_size, replacement=True)
            pt = pt_prob.gather(dim=0, index=steps) * len(pt_prob)
        elif method == 'uniform':
            steps = torch.randint(low=0, high=self.steps, size=(batch_size,)).long()
            pt = torch.ones_like(steps).float()
        else:
            raise ValueError
        
        return steps.to(self.device), pt.to(self.device)

    def get_noised_interaction(self, x_0, t):

        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None]
        mean_ = sqrt_alpha_bar * x_0
        std_ = self.sqrt_one_minus_alpha_bar[t][:, None]
        noise = torch.randn_like(x_0)

        noised_interaction = mean_ + std_ * noise # reparmeter
        return noised_interaction, noise

    def sample_new_interaction(self, model, x_0, div, guide_w, sampling_steps: int, sampling_noise=False):

        assert sampling_steps <= self.steps, "Too much steps in inference."

        batch_size = x_0.shape[0]

        if sampling_steps == 0:
            x_t = x_0
        else:
            # T = [steps, ... , steps], len(T) = batch size
            T = torch.tensor([sampling_steps] * x_0.shape[0]).to(x_0.device)
            x_t, _noise = self.get_noised_interaction(x_0, T)

        reverse_t = list(range(self.steps))[::-1] * 2
        div_mask = torch.cat((torch.zeros(div.shape[0]), torch.ones(div.shape[0])), dim=0).to(x_0.device) 
        div = torch.cat([div] * 2, dim=0)
        x_0 = torch.cat([x_0] * 2, dim=0)
        
        if self.noise_scale == 0:
            for t_idx in reverse_t:
                x_t = torch.cat([x_t] * 2, dim=0)
                t = torch.tensor([t_idx] * x_t.shape[0]).to(x_0.device) # Shape: (batch_size, )
                x_t = model(x_t, t, div, div_mask)
                x_t = (1 + guide_w) * x_t[:batch_size] - guide_w * x_t[batch_size:]
        else:
            for t_idx in reverse_t:
                x_t = torch.cat([x_t] * 2, dim=0)
                t = torch.tensor([t_idx] * x_t.shape[0]).to(x_0.device)
                x_0_hat = model(x_t, t, div, div_mask)
                x_0_hat = (1 + guide_w) * x_0_hat[:batch_size] - guide_w * x_0_hat[batch_size:]
                mean_hat = self.posterior_mean_coef1[t[:batch_size]][:, None] * x_t[:batch_size] + self.posterior_mean_coef2[t[:batch_size]][:, None] * x_0_hat
                if sampling_noise is True:
                    variance = self.posterior_log_variance_clipped[t[:batch_size]][:, None]
                    if t_idx > 0:
                        noise = torch.randn_like(x_t[:batch_size])
                    else:
                        noise = torch.zeros_like(x_t[:batch_size])
                    x_t = mean_hat + torch.exp(0.5 * variance) * noise
                else:
                    x_t = mean_hat
        return x_t
    
    def SNR(self, t):
        # Compute the signal-to-noise ratio for a single timestep.
        self.alpha_bar = self.alpha_bar.to(t.device)
        return self.alpha_bar[t] / (1 - self.alpha_bar[t])


class D3Rec(nn.Module):
    def __init__(self, dims, n_item, n_cate, dim_step=10, dropout=0.5):
        super(D3Rec, self).__init__()

        self.pos_enc_layer = PosEmb(dim_step)
        self.mlp = MyMLP(n_item, n_cate, dims, dim_step, dropout)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data, mean=0.0, std=0.001)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x_t, timesteps, probs, probs_mask):
        time_emb = self.pos_enc_layer(timesteps)
        latent = torch.cat([x_t, time_emb], dim=-1)
        return self.mlp(latent, probs, probs_mask)
