import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb
import torch.fft
import os
import matplotlib.pyplot as plt
import math

from config.locomotion_config import Config
import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class HaarWaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # Haar wavelet filters
        self.low_pass_filter = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32) / np.sqrt(2), requires_grad=False)
        self.high_pass_filter = nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32) / np.sqrt(2), requires_grad=False)

    def forward(self, x):
        device = x.device
        low_pass_filter = self.low_pass_filter.to(device)
        high_pass_filter = self.high_pass_filter.to(device)
        ori_dim = 3
        if x.ndim == 2:
            x = x.unsqueeze(1)
            ori_dim = 2
        batch, horizon, dim = x.shape
        x = x.permute(0, 2, 1).reshape(batch * dim, 1, horizon)  # (batch * dim, 1, horizon)

        # Apply low-pass and high-pass filters
        low_freq = nn.functional.conv1d(x, low_pass_filter.view(1, 1, -1), stride=2, padding=1)
        high_freq = nn.functional.conv1d(x, high_pass_filter.view(1, 1, -1), stride=2, padding=1)

        low_freq = low_freq.view(batch, dim, -1).permute(0, 2, 1)  # (batch, horizon//2, dim)
        high_freq = high_freq.view(batch, dim, -1).permute(0, 2, 1)  # (batch, horizon//2, dim)
        if ori_dim == 2:
            low_freq = low_freq.squeeze(1)            
            high_freq = high_freq.squeeze(1)

        return low_freq, high_freq

class HaarWaveletInverseTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # Haar wavelet filters
        self.low_pass_filter = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32) / np.sqrt(2), requires_grad=False)
        self.high_pass_filter = nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32) / np.sqrt(2), requires_grad=False)

    def forward(self, low_freq, high_freq):
        device = low_freq.device
        low_pass_filter = self.low_pass_filter.to(device)
        high_pass_filter = self.high_pass_filter.to(device)

        batch, horizon, dim = low_freq.shape

        low_freq = low_freq.permute(0, 2, 1).reshape(batch * dim, 1, horizon)
        high_freq = high_freq.permute(0, 2, 1).reshape(batch * dim, 1, horizon)

        # Upsample and apply inverse filters
        low_upsampled = nn.functional.conv_transpose1d(
            low_freq, low_pass_filter.view(1, 1, -1), stride=2, padding=1
        )
        high_upsampled = nn.functional.conv_transpose1d(
            high_freq, high_pass_filter.view(1, 1, -1), stride=2, padding=1
        )

        reconstructed = low_upsampled + high_upsampled
        reconstructed = reconstructed.view(batch, dim, -1).permute(0, 2, 1)  # (batch, horizon, dim)

        return reconstructed

act_fn = nn.ReLU()

def batch_dft(input_batch):
    dft_result = torch.fft.fft(input_batch, dim=1)
    return dft_result

def batch_idft(dft_transformed):
    return torch.fft.ifft(dft_transformed, dim=1).real  

class MLP_for_fourier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_for_fourier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class DFT_MLP_IDFT(nn.Module):
    def __init__(self, dimension, hidden_dim):
        super(DFT_MLP_IDFT, self).__init__()
        self.magnitude_mlp = MLP_for_fourier(dimension, hidden_dim, dimension)  
        self.phase_mlp = MLP_for_fourier(dimension, hidden_dim, dimension)      
    def forward(self, x):        
        dft_result = batch_dft(x)        
        magnitude = torch.abs(dft_result) 
        phase = torch.angle(dft_result)

        new_magnitude = self.magnitude_mlp(magnitude)
        new_phase = self.phase_mlp(phase)

        real_part = new_magnitude * torch.cos(new_phase)
        imag_part = new_magnitude * torch.sin(new_phase)        
        dft_transformed = torch.complex(real_part, imag_part)

        output = batch_idft(dft_transformed) + x  # (batch_size, horizon, dimension)
        return output

class Cross_Fourier_Fusion_Conditioner(nn.Module):
    def __init__(self, dim, num_heads, obs_dim):
        super(Cross_Fourier_Fusion_Conditioner, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        self.fourier_high = DFT_MLP_IDFT(obs_dim, dim*4)
        self.fourier_low = DFT_MLP_IDFT(obs_dim, dim*4)       
        
        self.query_proj = nn.Sequential(                        
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
        
        self.key_proj = nn.Sequential(                        
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )       
        self.value_proj = nn.Sequential(                        
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    ) 
        self.low_mlp = nn.Sequential(
                        nn.Linear(obs_dim, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
        
        self.high_mlp = nn.Sequential(
                        nn.Linear(obs_dim, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )  
              
    def split_heads(self, x):        
        batch_size, _, horizon = x.size()
        x = x.view(batch_size, self.num_heads, horizon,self.head_dim)
        return x.permute(0, 2, 1, 3)  

    def forward(self, high_freq, low_freq):
        high_freq_1 = self.fourier_high(high_freq)
        low_freq_1 = self.fourier_low(low_freq)        
        high_freq_1 = self.high_mlp(high_freq_1)
        low_freq_1 = self.low_mlp(low_freq_1)       
        # pdb.set_trace()
        Q = self.query_proj(high_freq_1)           
        K = self.key_proj(low_freq_1)    
        V_high = self.value_proj(high_freq_1)  
        V_low = self.value_proj(low_freq_1)   
        
        Q = self.split_heads(Q.transpose(1,2))  # (batch_size, num_heads, horizon, head_dim)
        K = self.split_heads(K.transpose(1,2))  # (batch_size, num_heads, horizon, head_dim)
        V_high = self.split_heads(V_high.transpose(1,2))  # (batch_size, num_heads, horizon, head_dim)
        V_low = self.split_heads(V_low.transpose(1,2))    # (batch_size, num_heads, horizon, head_dim)
        # pdb.set_trace()

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, horizon, horizon)
        attention_scores = attention_scores/ math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)    # softmax

        output_high = torch.matmul(attention_probs, V_high)  # (batch_size, num_heads, horizon, head_dim)
        output_low = torch.matmul(attention_probs, V_low)    # (batch_size, num_heads, horizon, head_dim)
       
        output_high = output_high.permute(0, 2, 1, 3).contiguous()  # (batch_size, horizon, num_heads, head_dim)
        output_low = output_low.permute(0, 2, 1, 3).contiguous()    # (batch_size, horizon, num_heads, head_dim)

        return output_high, output_low
      
class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        if self.predict_epsilon:
            # Cause we condition on obs at t=0
            noise[:, 0, self.action_dim:] = 0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        if self.model.calc_energy:
            assert self.predict_epsilon
            x_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model_low, model_high, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1, ar_inv=False, train_only_inv=False):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model_low = model_low
        self.model_high = model_high
        self.horizon_for_transform = horizon//2 + 1
        self.transform = HaarWaveletTransform()
        self.inv_transform = HaarWaveletInverseTransform()
        self.cross_attention = Cross_Fourier_Fusion_Conditioner(dim=128, num_heads=1, obs_dim=observation_dim)    
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)       

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
       
        discounts = discount ** torch.arange(self.horizon_for_transform, dtype=torch.float)
       
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights
    
    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_low,x_high, cond_low, cond_high, cross_low, cross_high, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            # edit
            epsilon_cond_low = self.model_low(x_low, cond_low, cross_low, t, returns, use_dropout=False)
            epsilon_uncond_low = self.model_low(x_low, cond_low, cross_low, t, returns, force_dropout=True)
            epsilon_low = epsilon_uncond_low + self.condition_guidance_w*(epsilon_cond_low - epsilon_uncond_low)
            epsilon_cond_high = self.model_high(x_high, cond_high, cross_high, t, returns, use_dropout=False)
            epsilon_uncond_high = self.model_high(x_high, cond_high, cross_high, t, returns, force_dropout=True)
            epsilon_high = epsilon_uncond_high + self.condition_guidance_w*(epsilon_cond_high - epsilon_uncond_high)
            
        #     epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
        #     epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
        #     epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        # else:
        #     epsilon = self.model(x, cond, t)
            # end

        t = t.detach().to(torch.int64)
        # edit
        x_recon_low = self.predict_start_from_noise(x_low, t=t, noise=epsilon_low)
        x_recon_high = self.predict_start_from_noise(x_high, t=t, noise=epsilon_high)

        if self.clip_denoised:
            x_recon_low.clamp_(-1., 1.)
            x_recon_high.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean_low, posterior_variance_low, posterior_log_variance_low = self.q_posterior(
                x_start=x_recon_low, x_t=x_low, t=t)
        model_mean_high, posterior_variance_high, posterior_log_variance_high = self.q_posterior(
            x_start=x_recon_high, x_t=x_high, t=t)
        return model_mean_low, model_mean_high,posterior_variance_low, posterior_variance_high, posterior_log_variance_low, posterior_log_variance_high
        # end
    @torch.no_grad()
    def p_sample(self, x_low, x_high, cond_low, cond_high, cross_low, cross_high, t, returns=None):
        
        b, *_, device = *x_low.shape, x_low.device
        model_mean_low, model_mean_high, _, _, model_log_variance_low, model_log_variance_high = self.p_mean_variance(
            x_low = x_low, x_high = x_high, cond_low = cond_low,
             cond_high = cond_high, cross_low = cross_low, cross_high = cross_high, t=t, returns=returns)
        noise_low = 0.5*torch.randn_like(x_low)
        noise_high = 0.5*torch.randn_like(x_high)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_low.shape) - 1)))
        return_low = model_mean_low + nonzero_mask * (0.5 * model_log_variance_low).exp() * noise_low
        return_high = model_mean_high + nonzero_mask * (0.5 * model_log_variance_high).exp() * noise_high
        return return_low, return_high
       

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        shape = (batch_size, self.horizon_for_transform, shape[2])        
        x_low = 0.5*torch.randn(shape, device=device)
        x_high = 0.5*torch.randn(shape, device=device)
        cond_low, cond_high = self.get_cond_low_and_high(cond)
        x_low = apply_conditioning(x_low, cond_low, 0)
        x_high = apply_conditioning(x_high, cond_high, 0)
        cross_high, cross_low = self.cross_attention(cond_high[0].unsqueeze(1), cond_low[0].unsqueeze(1))
        cross_high = cross_high.squeeze(1).squeeze(1)
        cross_low = cross_low.squeeze(1).squeeze(1) 
        # if return_diffusion: diffusion = [x]
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)           
            x_low, x_high = self.p_sample(x_low, x_high, cond_low, cond_high, cross_low, cross_high,
                                          timesteps, returns)
            x_low = apply_conditioning(x_low, cond_low, 0)
            x_high = apply_conditioning(x_high, cond_high, 0)
          
            progress.update({'t': i})

            # if return_diffusion: diffusion.append(x)

        progress.close()
       
        x = self.inv_transform(x_low, x_high)
        return x
        

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

 
    def get_cond_low_and_high(self, cond):
        con = cond[0]
        con_low, con_high = self.transform(con)        
        cond_low = {0: con_low}
        cond_high = {0: con_high}
        return cond_low, cond_high    
        
    def p_losses(self, x_start, cond, t, returns=None):
          
        # noise = torch.randn_like(x_start)
        x_start_low, x_start_high = self.transform(x_start)               
        noise_low = torch.randn_like(x_start_low)
        noise_high = torch.randn_like(x_start_high)
           

        # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # x_noisy = apply_conditioning(x_noisy, cond, 0)
        x_noisy_low = self.q_sample(x_start=x_start_low, t=t, noise=noise_low)
        x_noisy_high = self.q_sample(x_start=x_start_high, t=t, noise=noise_high)
        cond_low, cond_high = self.get_cond_low_and_high(cond)        
        x_noisy_low = apply_conditioning(x_noisy_low, cond_low, 0)
        x_noisy_high = apply_conditioning(x_noisy_high, cond_high, 0)
        cross_high, cross_low = self.cross_attention(cond_high[0].unsqueeze(1), cond_low[0].unsqueeze(1))
        cross_high = cross_high.squeeze(1).squeeze(1)
        cross_low = cross_low.squeeze(1).squeeze(1)        

        x_recon_low = self.model_low(x_noisy_low, cond, cross_low, t, returns)
        x_recon_high = self.model_high(x_noisy_high, cond, cross_high, t, returns)        

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise_low.shape == x_recon_low.shape        

        if self.predict_epsilon:            
            loss_low, info_low = self.loss_fn(x_recon_low, noise_low)
            loss_high, info_high = self.loss_fn(x_recon_high, noise_high)
            
            loss = 2 * (loss_low * loss_low/ (loss_low + loss_high) + loss_high * loss_high/ (loss_low + loss_high))            
            info = {'a0_loss_low':info_low['a0_loss'],
                    'a0_loss_high':info_high['a0_loss']}            
        else:
            loss, info = self.loss_fn(x_recon, x_start)       

        return loss, info

    def loss(self, x, cond, returns=None, steps=None):
        if self.train_only_inv:
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss = self.inv_model.calc_loss(x_comb_t, a_t)
                info = {'a0_loss':loss}
            else:
                pred_a_t = self.inv_model(x_comb_t)
                loss = F.mse_loss(pred_a_t, a_t)
                info = {'a0_loss': loss}
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns)
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
            else:
                pred_a_t = self.inv_model(x_comb_t)
                inv_loss = F.mse_loss(pred_a_t, a_t)

            loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


class ARInvModel(nn.Module):
    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1))
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                  self.low_act + (l_i + 1) * self.bin_size).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                     l_action[:, i])

        return loss/self.action_dim


class ActionGaussianDiffusion(nn.Module):
    # Assumes horizon=1
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, action_start, state, t, returns=None):
        noise = torch.randn_like(action_start)
        action_noisy = self.q_sample(x_start=action_start, t=t, noise=noise)

        if self.model.calc_energy:
            assert self.predict_epsilon
            action_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        pred = self.model(action_noisy, state, t, returns)

        assert noise.shape == pred.shape

        if self.predict_epsilon:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.mse_loss(pred, action_start)

        return loss, {'a0_loss':loss}

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        assert x.shape[1] == 1 # Assumes horizon=1
        x = x[:,0,:]
        cond = x[:,self.action_dim:] # Observation
        x = x[:,:self.action_dim] # Action
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

