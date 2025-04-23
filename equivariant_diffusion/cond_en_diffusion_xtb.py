from equivariant_diffusion import utils
import numpy as np
import math
import torch
import subprocess
from tqdm import tqdm
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
import os
import shutil
import pickle
import json
import warnings
import multiprocessing
import torch.nn as nn
from qm9.property_prediction import prop_utils
from functorch import vmap, jacrev

try:
    from xtb.libxtb import VERBOSITY_MINIMAL, VERBOSITY_MUTED
    from xtb.interface import Calculator, Param
except Exception as e:
    print("Cannot import XTB-Python")
    print(e)

warnings.simplefilter("ignore")

ATOM_NUMBER_TO_TYPE_DICT = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
                            11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 20: 'Ca',
                            21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
                            30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr',
                            39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag',
                            48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba',
                            57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb',
                            66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
                            75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi',
                            84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
                            93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md',
                            102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',
                            110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts',
                            118: 'Og'}


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma ** 2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
        (
                torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8)
                + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2)
                - 0.5
        ) * node_mask
    )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu) ** 2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (d * torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8)
            + 0.5 * (d * q_sigma ** 2 + mu_norm2) / (p_sigma ** 2)
            - 0.5 * d
            )


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)
        # gamma = -log[alpha2/(1-alpha2)]

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""

    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class EnVariationalDiffusion(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """

    def __init__(
            self,
            dynamics: models.EGNN_dynamics_QM9, in_node_nf: int, n_dims: int,
            timesteps: int = 1000, parametrization='eps', noise_schedule='learned',
            noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.), include_charges=True):
        super().__init__()

        # variables for efficient guidance
        self.last_property_MAE = None
        self.every_n_step = None
        self.use_hist_grad = None
        self.hist_grad_force = None
        self.hist_grad_property = None
        self.save_path = None  # for saving intermediate results

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.include_charges = include_charges
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        if noise_schedule != 'learned':
            self.check_issues_norm_values()

        self.args = None

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def phi(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_int = z[:, :, self.n_dims + self.num_classes:self.n_dims + self.num_classes + 1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)  # (bs, 1, 1)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)  # (bs, 1, 1)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error

    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h

    # ChemGuide
    def cond_sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False,
                                  cond_fn=None, guidance_kwargs=None, delta_z0=None):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        if delta_z0 is not None:
            alpha_t = self.alpha(gamma_0, target_tensor=net_out)
            # eq (9), https://arxiv.org/pdf/2302.07121
            net_out = net_out - alpha_t / torch.sqrt(1 - alpha_t * alpha_t) * delta_z0

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)

        xh = self.cond_sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise,
                                     cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)

        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    # ChemGuide
    def cond_sample_normal(self, mu, sigma, node_mask,
                           fix_noise=False, cond_fn=None, guidance_kwargs=None):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)

        cond_mu = self.cond_mean(mu=mu, sigma=sigma, cond_fn=cond_fn,
                                 guidance_kwargs=guidance_kwargs)

        return cond_mu + sigma * eps

    # ChemGuide
    def cond_mean(self, mu, sigma, cond_fn, guidance_kwargs):
        if cond_fn is None:  # no conditional guidance
            new_mu = mu
        else:  # add guidance
            gradient = cond_fn(**guidance_kwargs)
            new_mu = mu + sigma * gradient
        return new_mu

    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_t.
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)

        # Compute delta indicator masks.
        h_integer = torch.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""
        # x, h is the encoded (by AE) z
        # context is None when doing normal training (i.e., unconditional generation)
        # when self.training is True, t0_always is False (see in the implementation of "forward")

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()  # draw from [lowest_t, T+1)
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps  # Line 19 of Algorithm 1; also see the line under equation (3)

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)

        # Neural net prediction.
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)
        # in the implementation of self.compute_error, gamma_t is not used

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        # TODO: here z1 means zT (the start of the denoising process)
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        return zs

    # ChemGuide
    def cond_sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False,
                                  cond_fn=None, guidance_kwargs=None, delta_z0=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)
        if delta_z0 is not None:
            alpha_t = self.alpha(gamma_t, target_tensor=eps_t)
            # eq (9), https://arxiv.org/pdf/2302.07121
            eps_t = eps_t - alpha_t / torch.sqrt(1 - alpha_t * alpha_t) * delta_z0

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the parameters derived from zt.
        zs = self.cond_sample_normal(mu, sigma, node_mask, fix_noise,
                                     cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )

        return zs, alpha_t_given_s

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        # (B, max_node, num_classes)
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        # (B, max_node, num_classes + 1)

        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z

    @torch.no_grad()  # EnVariationalDiffusion
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        """
        # if noise is fixed, then it will be used for all n_samples
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)  # z_T
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    # ChemGuide for oracle guidance, noisy guidance, and clean guidance
    @torch.no_grad()
    def cond_sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False,
                    cond_fn=None, guidance_kwargs=None, save_traj_flag=False, args=None):
        """
        Draw (conditional) samples from the generative model with guidance.
        node_mask B = 100
        n_samples = 100
        """
        self.args = args
        # if noise is fixed, then it will be used for all n_samples
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        prev_h0, prev_x0 = None, None
        guidance_steps = guidance_kwargs["guidance_steps"]
        scale = guidance_kwargs["scale"]
        grad_clip_threshold = guidance_kwargs["grad_clip_threshold"]
        guidance_property = guidance_kwargs["property"]

        # init the new variables for efficient guidance here
        self.every_n_step = guidance_kwargs["every_n_step"]
        self.use_hist_grad = guidance_kwargs["use_hist_grad"]
        self.hist_grad_force = 0
        self.hist_grad_property = 0
        recurrent_times = guidance_kwargs["recurrent_times"]

        self.save_path = f'eval/every_{self.every_n_step}_step_use_hist_grad_{self.use_hist_grad}_' \
                         f'grad_clip{grad_clip_threshold}_guidance_steps_{guidance_steps}_scale_' \
                         f'{scale}_package_{self.args.package}_property_{guidance_property}_rt_' \
                         f'{recurrent_times}_use_neural_{self.args.use_neural_guidance}_use_cond_xtb_' \
                         f'{self.args.use_cond_geo}/seed{self.args.seed}'

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass

        if 'prop' in self.args.package:
            get_force_fn = self.get_property_from_command_xtb
        else:
            get_force_fn = self.get_force_from_command_xtb

        del guidance_kwargs["max_core"]
        guidance_kwargs["get_force_fn"] = get_force_fn

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(reversed(range(0, self.T))):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            helper_kwargs = {'t': s + 1, 'mu': None, 'zt': z, 'n_samples': n_samples, 'node_mask': node_mask,
                             'edge_mask': edge_mask, 'context': context, 'fix_noise': fix_noise}
            guidance_kwargs.update(helper_kwargs)

            guidance_kwargs["scale"] = scale
            z, alpha_t_given_alpa_s = self.cond_sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context,
                                                                     fix_noise=fix_noise, cond_fn=cond_fn,
                                                                     guidance_kwargs=guidance_kwargs)
            if (s + 1) <= guidance_steps and scale != 0:
                for recurrent_step in range(recurrent_times - 1):
                    # after the optimization (either force/prop guidance) is performed once
                    # apply scale decay here
                    guidance_kwargs["scale"] = max(scale / np.power(10, (recurrent_step + 1)), 1.0)
                    # project z_{t-1} back to zt, universal guidance, eq (10), https://arxiv.org/pdf/2302.07121
                    z = alpha_t_given_alpa_s * z
                    # z_{t-1}--->z_t
                    z, _ = self.cond_sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context,
                                                          fix_noise=fix_noise,
                                                          cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)

            if save_traj_flag:
                z0_x, z0_h = self.one_step_sample_xh_given_zt(n_samples=n_samples, zt=z.clone(), timestep=s,
                                                              node_mask=node_mask, edge_mask=edge_mask,
                                                              context=context, fix_noise=fix_noise)
                z0_xh = torch.cat([z0_x, z0_h['categorical'], z0_h['integer']], dim=2)
                diffusion_utils.assert_correctly_masked(z0_xh, node_mask)
                x0, h0 = self.vae.decode(z0_xh, node_mask, edge_mask, context)
                file = open(f"traj_scale{scale}_last{guidance_steps}.txt", "a+")
                # Saving the array in a text file
                h0_change_text = ""
                x0_change_text = ""
                if prev_x0 is not None:
                    if not torch.equal(prev_h0, h0['categorical'][0]):
                        print(f"h0 changes")
                        h0_change_text = "change"
                    if not torch.equal(prev_x0, x0[0]):
                        print(f"x0 changes")
                        x0_change_text = "change"
                content = f"Step {s}\n" + f"\th0 {h0_change_text}: " + str(h0['categorical'][0]) + \
                          f"\n\tx0 {x0_change_text}: " + str(x0[0]) + "\n\n"
                file.write(content)
                file.close()
                prev_h0, prev_x0 = h0['categorical'][0], x0[0]

        # Finally sample p(x, h | z_0).
        helper_kwargs = {'t': 0, 'mu': None, 'zt': z, 'n_samples': n_samples, 'node_mask': node_mask,
                         'edge_mask': edge_mask, 'context': context, 'fix_noise': fix_noise}
        guidance_kwargs.update(helper_kwargs)
        x, h = self.cond_sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise,
                                              cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    # ChemGuide for evolutionary algorithm, used in appendix
    @torch.no_grad()
    def cond_sample_evolutionary(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False,
                                 cond_fn=None, guidance_kwargs=None, save_traj_flag=False, args=None):
        """
        Draw (conditional) samples from the generative model with guidance.
        node_mask B = 100
        n_samples = 100
        """
        self.args = args
        # if noise is fixed, then it will be used for all n_samples
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        z_list = [z]

        guidance_steps = guidance_kwargs["guidance_steps"]
        scale = guidance_kwargs["scale"]
        grad_clip_threshold = guidance_kwargs["grad_clip_threshold"]
        guidance_property = guidance_kwargs["property"]

        # init the new variables for efficient guidance here
        self.every_n_step = guidance_kwargs["every_n_step"]
        self.use_hist_grad = guidance_kwargs["use_hist_grad"]
        self.hist_grad_force = 0
        self.hist_grad_property = 0
        recurrent_times = guidance_kwargs["recurrent_times"]

        self.save_path = f'eval/every_{self.every_n_step}_step_use_hist_grad_{self.use_hist_grad}_' \
                         f'grad_clip{grad_clip_threshold}_guidance_steps_{guidance_steps}_scale_' \
                         f'{scale}_package_{self.args.package}_property_{guidance_property}_rt_' \
                         f'{recurrent_times}_use_neural_{self.args.use_neural_guidance}_use_cond_xtb_' \
                         f'{self.args.use_cond_geo}_evo_num_beams{self.args.num_beams}_interval{self.args.check_variants_interval}/seed{self.args.seed}'

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass

        if 'prop' in self.args.package:
            get_force_fn = self.get_property_from_command_xtb
        else:
            get_force_fn = self.get_force_from_command_xtb

        del guidance_kwargs["max_core"]
        guidance_kwargs["get_force_fn"] = get_force_fn

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(reversed(range(0, self.T))):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            if (s + 1) % self.args.check_variants_interval == 0:
                print(f'adding variants at step={s + 1}')
                z_best = z_list[0]
                for _ in range(self.args.num_beams - 1):
                    if fix_noise:
                        # Noise is broadcast over the batch axis, useful for visualizations.
                        variant = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
                    else:
                        variant = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

                    variant = variant * 1e-1 + z_best  # [for variants]

                    z_list.append(variant)

            variants_list = []

            for z_vt in z_list:
                helper_kwargs = {'t': s + 1, 'mu': None, 'zt': z_vt, 'n_samples': n_samples, 'node_mask': node_mask,
                                 'edge_mask': edge_mask, 'context': context, 'fix_noise': fix_noise}
                guidance_kwargs.update(helper_kwargs)

                guidance_kwargs["scale"] = scale
                z_vt, alpha_t_given_alpa_s = self.cond_sample_p_zs_given_zt(s_array, t_array, z_vt, node_mask,
                                                                            edge_mask, context,
                                                                            fix_noise=fix_noise, cond_fn=cond_fn,
                                                                            guidance_kwargs=guidance_kwargs)
                variants_list.append(z_vt)

            z_list = variants_list

            if (s + 1) % self.args.check_variants_interval == 0:  # prune the variants
                guidance_kwargs['t'] = s
                z_list = self.select_best_evolution(z_list, **guidance_kwargs)  # choose the best beam
                print(f'variants pruned at step={s + 1}')

        assert len(z_list) == 1  # there should still be only 1 candidate
        z = z_list[0]

        # Finally sample p(x, h | z_0).
        helper_kwargs = {'t': 0, 'mu': None, 'zt': z, 'n_samples': n_samples, 'node_mask': node_mask,
                         'edge_mask': edge_mask, 'context': context, 'fix_noise': fix_noise}
        guidance_kwargs.update(helper_kwargs)
        x, h = self.cond_sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise,
                                              cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    # ChemGuide for bilevel optimization with noisy guidance
    @torch.no_grad()
    def cond_sample_bilevel_noisy(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False,
                                  cond_force_fn=None,
                                  cond_property_fn=None, guidance_kwargs=None, save_traj_flag=False, args=None):
        """
        Draw (conditional) samples from the generative model with guidance.
        node_mask B = 100
        n_samples = 100
        """
        self.args = args
        # if noise is fixed, then it will be used for all n_samples
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        prev_h0, prev_x0 = None, None
        guidance_steps = guidance_kwargs["guidance_steps"]
        scale = guidance_kwargs["scale"]
        grad_clip_threshold = guidance_kwargs["grad_clip_threshold"]
        chemistry_package = guidance_kwargs["package"]
        guidance_property = guidance_kwargs["property"]

        # init the new variables for efficient guidance here
        self.every_n_step = guidance_kwargs["every_n_step"]
        self.use_hist_grad = guidance_kwargs["use_hist_grad"]
        self.hist_grad_force = 0
        self.hist_grad_property = 0
        recurrent_times = guidance_kwargs["recurrent_times"]
        clf_scale_force = guidance_kwargs["clf_scale_force"]
        clf_scale_prop = guidance_kwargs["clf_scale_prop"]

        get_force_fn = self.get_force_from_command_xtb

        del guidance_kwargs["max_core"]
        guidance_kwargs["get_force_fn"] = get_force_fn

        self.save_path = f'eval/every_{self.every_n_step}_step_grad_clip{grad_clip_threshold}_guidance_steps_{guidance_steps}_fs_{clf_scale_force}_ps_{clf_scale_prop}_package_{chemistry_package}_property_{guidance_property}_rt_{recurrent_times}_bilevel/seed{self.args.seed}'

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(reversed(range(0, self.T))):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            helper_kwargs = {'t': s + 1, 'mu': None, 'zt': z, 'n_samples': n_samples, 'node_mask': node_mask,
                             'edge_mask': edge_mask, 'context': context, 'fix_noise': fix_noise}
            guidance_kwargs.update(helper_kwargs)

            if (s + 1) <= guidance_steps:
                # inner objective:
                # perform recurrent neural property guidance (would be fast with autograd)

                if (s + 1) % guidance_kwargs["every_n_step"] == 0:
                    self.every_n_step = guidance_kwargs["every_n_step"]
                    for recurrent_step in range(recurrent_times):
                        print('recurrent step called!')
                        # zt--->z_{t-1}
                        guidance_kwargs["scale"] = max(clf_scale_prop / np.power(10, recurrent_step), 1.0)
                        # clf_scale_prop

                        z, alpha_t_given_alpa_s = self.cond_sample_p_zs_given_zt(s_array, t_array, z, node_mask,
                                                                                 edge_mask, context,
                                                                                 fix_noise=fix_noise,
                                                                                 cond_fn=cond_property_fn,
                                                                                 guidance_kwargs=guidance_kwargs)

                        # project z_{t-1} back to zt, universal guidance, eq (10), https://arxiv.org/pdf/2302.07121
                        z = alpha_t_given_alpa_s * z

                # outer objective
                # perform force guidance (from non-differentiable chemistry package)
                guidance_kwargs["scale"] = clf_scale_force
                self.every_n_step = 1  # always add xtb guidance
                z, _ = self.cond_sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context,
                                                      fix_noise=fix_noise, cond_fn=cond_force_fn,
                                                      guidance_kwargs=guidance_kwargs)
                if save_traj_flag:
                    z0_x, z0_h = self.one_step_sample_xh_given_zt(n_samples=n_samples, zt=z.clone(), timestep=s,
                                                                  node_mask=node_mask, edge_mask=edge_mask,
                                                                  context=context, fix_noise=fix_noise)
                    z0_xh = torch.cat([z0_x, z0_h['categorical'], z0_h['integer']], dim=2)
                    diffusion_utils.assert_correctly_masked(z0_xh, node_mask)
                    x0, h0 = self.vae.decode(z0_xh, node_mask, edge_mask, context)
                    file = open(f"traj_scale{scale}_last{guidance_steps}.txt", "a+")
                    # Saving the array in a text file
                    h0_change_text = ""
                    x0_change_text = ""
                    if prev_x0 is not None:
                        if not torch.equal(prev_h0, h0['categorical'][0]):
                            print(f"h0 changes")
                            h0_change_text = "change"
                        if not torch.equal(prev_x0, x0[0]):
                            print(f"x0 changes")
                            x0_change_text = "change"
                    content = f"Step {s}\n" + f"\th0 {h0_change_text}: " + str(h0['categorical'][0]) + \
                              f"\n\tx0 {x0_change_text}: " + str(x0[0]) + "\n\n"
                    file.write(content)
                    file.close()
                    prev_h0, prev_x0 = h0['categorical'][0], x0[0]
            else:
                z, alpha_t_given_alpa_s = self.cond_sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask,
                                                                         context,
                                                                         fix_noise=fix_noise, cond_fn=cond_force_fn,
                                                                         guidance_kwargs=guidance_kwargs)

        # Finally sample p(x, h | z_0).
        helper_kwargs = {'t': 0, 'mu': None, 'zt': z, 'n_samples': n_samples, 'node_mask': node_mask,
                         'edge_mask': edge_mask, 'context': context, 'fix_noise': fix_noise}
        guidance_kwargs.update(helper_kwargs)
        guidance_kwargs["scale"] = clf_scale_prop
        self.every_n_step = 1  # always add guidance
        x, h = self.cond_sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise,
                                              cond_fn=cond_property_fn, guidance_kwargs=guidance_kwargs)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    # ChemGuide for bilevel optimization with clean guidance
    @torch.no_grad()
    def cond_sample_bilevel_clean(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False,
                                  cond_force_fn=None, cond_property_fn=None, guidance_kwargs=None,
                                  save_traj_flag=False, args=None):
        """
        Draw (conditional) samples from the generative model with guidance.
        node_mask B = 100
        n_samples = 100
        """
        self.args = args
        # if noise is fixed, then it will be used for all n_samples
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        curr_h0, curr_x0 = None, None
        guidance_steps = guidance_kwargs["guidance_steps"]
        scale = guidance_kwargs["scale"]
        grad_clip_threshold = guidance_kwargs["grad_clip_threshold"]
        chemistry_package = guidance_kwargs["package"]
        guidance_property = guidance_kwargs["property"]

        # init the new variables for efficient guidance here
        self.every_n_step = guidance_kwargs["every_n_step"]
        self.use_hist_grad = guidance_kwargs["use_hist_grad"]
        self.hist_grad_force = 0
        self.hist_grad_property = 0
        recurrent_times = guidance_kwargs["recurrent_times"]
        clf_scale_force = guidance_kwargs["clf_scale_force"]
        clf_scale_prop = guidance_kwargs["clf_scale_prop"]

        get_force_fn = self.get_force_from_command_xtb

        del guidance_kwargs["max_core"]
        guidance_kwargs["get_force_fn"] = get_force_fn

        self.save_path = f'eval/grad_clip{grad_clip_threshold}_guidance_steps_{guidance_steps}_fs_{clf_scale_force}_ps_{clf_scale_prop}_package_{chemistry_package}_property_{guidance_property}_rt_{recurrent_times}_bilevel_uni_guide/seed{self.args.seed}'

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(reversed(range(0, self.T))):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            helper_kwargs = {'t': s + 1, 'mu': None, 'zt': z, 'n_samples': n_samples, 'node_mask': node_mask,
                             'edge_mask': edge_mask, 'context': context, 'fix_noise': fix_noise}
            guidance_kwargs.update(helper_kwargs)

            if (s + 1) <= guidance_steps:
                # inner objective:
                # perform recurrent neural property guidance (would be fast with autograd)
                if (s + 1) % guidance_kwargs["every_n_step"] == 0:
                    z0_x, z0_h = self.one_step_sample_xh_given_zt(n_samples=n_samples, zt=z.clone(), timestep=s + 1,
                                                                  # input is zt, and thus timestep=s+1(=t)
                                                                  node_mask=node_mask, edge_mask=edge_mask,
                                                                  context=context, fix_noise=fix_noise)
                    z0_xh = torch.cat([z0_x, z0_h['categorical'], z0_h['integer']], dim=2)
                    diffusion_utils.assert_correctly_masked(z0_xh, node_mask)

                    delta_z0_xh = torch.zeros_like(z0_xh)

                    # get the representation in clean data space
                    self.every_n_step = guidance_kwargs["every_n_step"]
                    for recurrent_step in range(recurrent_times):
                        print('recurrent step called!')
                        # zt--->z_{t-1}
                        guidance_kwargs["scale"] = clf_scale_prop
                        guidance_grad = cond_property_fn(step=recurrent_step, z0=z0_xh, delta_z0=delta_z0_xh,
                                                         **guidance_kwargs)
                        delta_z0_xh = delta_z0_xh + guidance_grad
                        # Project down to avoid numerical runaway of the center of gravity.
                        delta_z0_xh = torch.cat(
                            [diffusion_utils.remove_mean_with_mask(delta_z0_xh[:, :, :self.n_dims], node_mask),
                             delta_z0_xh[:, :, self.n_dims:]], dim=2)

                else:
                    delta_z0_xh = None  # do not optimize property

                # outer objective
                # perform force guidance (from non-differentiable chemistry package)
                guidance_kwargs["scale"] = clf_scale_force
                self.every_n_step = 1  # always add xtb guidance
                z, _ = self.cond_sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context,
                                                      fix_noise=fix_noise, cond_fn=cond_force_fn,
                                                      guidance_kwargs=guidance_kwargs, delta_z0=delta_z0_xh)
                # translate the delta in clean space back to zt (if delta_z0_xh is not None)
                if save_traj_flag:
                    z0_x, z0_h = self.one_step_sample_xh_given_zt(n_samples=n_samples, zt=z.clone(), timestep=s,
                                                                  node_mask=node_mask, edge_mask=edge_mask,
                                                                  context=context, fix_noise=fix_noise)
                    z0_xh = torch.cat([z0_x, z0_h['categorical'], z0_h['integer']], dim=2)
                    diffusion_utils.assert_correctly_masked(z0_xh, node_mask)
                    x0, h0 = self.vae.decode(z0_xh, node_mask, edge_mask, context)
                    file = open(f"traj_scale{scale}_last{guidance_steps}.txt", "a+")
                    # Saving the array in a text file
                    h0_change_text = ""
                    x0_change_text = ""
                    if curr_x0 is not None:
                        if not torch.equal(curr_h0, h0['categorical'][0]):
                            print(f"h0 changes")
                            h0_change_text = "change"
                        if not torch.equal(curr_x0, x0[0]):
                            print(f"x0 changes")
                            x0_change_text = "change"
                    content = f"Step {s}\n" + f"\th0 {h0_change_text}: " + str(h0['categorical'][0]) + \
                              f"\n\tx0 {x0_change_text}: " + str(x0[0]) + "\n\n"
                    file.write(content)
                    file.close()
                    curr_h0, curr_x0 = h0['categorical'][0], x0[0]
            else:  # no guidance
                z, alpha_t_given_alpa_s = self.cond_sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask,
                                                                         context,
                                                                         fix_noise=fix_noise, cond_fn=cond_force_fn,
                                                                         guidance_kwargs=guidance_kwargs)

        # Finally sample p(x, h | z_0).
        helper_kwargs = {'t': 0, 'mu': None, 'zt': z, 'n_samples': n_samples, 'node_mask': node_mask,
                         'edge_mask': edge_mask, 'context': context, 'fix_noise': fix_noise}
        guidance_kwargs.update(helper_kwargs)
        guidance_kwargs["scale"] = clf_scale_force
        self.every_n_step = 1  # always add xtb guidance
        x, h = self.cond_sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise,
                                              cond_fn=cond_force_fn, guidance_kwargs=guidance_kwargs)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    # ChemGuide
    @torch.no_grad()  # EnVariationalDiffusion
    def sample_xh_given_zt(self, n_samples, z, timestep, node_mask, edge_mask, context, fix_noise):
        """
        Draw samples from the generative model given zt
        """

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, timestep)):  # [0, timestep)
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)  # <= timestep-1
            t_array = s_array + 1  # <= timestep
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    # ChemGuide, using equation (3) in https://arxiv.org/pdf/2302.07121.pdf
    def one_step_sample_xh_given_zt(self, n_samples, zt, timestep, node_mask, edge_mask, context, fix_noise):
        """
        Draw samples from the generative model given zt
        """

        if timestep > 0:
            t_array = torch.full((n_samples, 1), fill_value=timestep, device=zt.device)
            t_array = t_array / self.T
            gamma_t = self.gamma(t_array)
            net_out = self.phi(zt, t_array, node_mask, edge_mask, context)
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            z0 = 1. / alpha_t * (zt - sigma_t * net_out)
        else:
            z0 = zt

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z0, node_mask, edge_mask, context, fix_noise=fix_noise)
        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context)

            diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        diffusion_utils.assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {
            'log_SNR_max': log_SNR_max.item(),
            'log_SNR_min': log_SNR_min.item()}
        print(info)

        return info


class EnHierarchicalVAE(torch.nn.Module):
    """
    The E(n) Hierarchical VAE Module.
    """

    def __init__(
            self,
            encoder: models.EGNN_encoder_QM9,
            decoder: models.EGNN_decoder_QM9,
            in_node_nf: int, n_dims: int, latent_node_nf: int,
            kl_weight: float,
            norm_values=(1., 1., 1.), norm_biases=(None, 0., 0.),
            include_charges=True):
        super().__init__()

        self.include_charges = include_charges

        self.encoder = encoder
        self.decoder = decoder

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.latent_node_nf = latent_node_nf
        self.num_classes = self.in_node_nf - self.include_charges
        self.kl_weight = kl_weight

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def compute_reconstruction_error(self, xh_rec, xh):
        """Computes reconstruction error."""

        bs, n_nodes, dims = xh.shape

        # Error on positions.
        x_rec = xh_rec[:, :, :self.n_dims]
        x = xh[:, :, :self.n_dims]
        error_x = sum_except_batch((x_rec - x) ** 2)

        # Error on classes.
        h_cat_rec = xh_rec[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat = xh[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat_rec = h_cat_rec.reshape(bs * n_nodes, self.num_classes)
        h_cat = h_cat.reshape(bs * n_nodes, self.num_classes)
        error_h_cat = F.cross_entropy(h_cat_rec, h_cat.argmax(dim=1), reduction='none')
        error_h_cat = error_h_cat.reshape(bs, n_nodes, 1)
        error_h_cat = sum_except_batch(error_h_cat)
        # error_h_cat = sum_except_batch((h_cat_rec - h_cat) ** 2)

        # Error on charges.
        if self.include_charges:
            h_int_rec = xh_rec[:, :, -self.include_charges:]
            h_int = xh[:, :, -self.include_charges:]
            error_h_int = sum_except_batch((h_int_rec - h_int) ** 2)
        else:
            error_h_int = 0.

        error = error_x + error_h_cat + error_h_int

        if self.training:
            denom = (self.n_dims + self.in_node_nf) * xh.shape[1]
            error = error / denom

        return error

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def compute_loss(self, x, h, node_mask, edge_mask, context):
        """Computes an estimator for the variational lower bound."""

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encode(x, h, node_mask, edge_mask, context)

        # KL distance.
        # KL for invariant features.
        zeros, ones = torch.zeros_like(z_h_mu), torch.ones_like(z_h_sigma)
        loss_kl_h = gaussian_KL(z_h_mu, ones, zeros, ones, node_mask)
        # KL for equivariant features.
        assert z_x_sigma.mean(dim=(1, 2), keepdim=True).expand_as(z_x_sigma).allclose(z_x_sigma, atol=1e-7)
        zeros, ones = torch.zeros_like(z_x_mu), torch.ones_like(z_x_sigma.mean(dim=(1, 2)))
        subspace_d = self.subspace_dimensionality(node_mask)
        loss_kl_x = gaussian_KL_for_dimension(z_x_mu, ones, zeros, ones, subspace_d)
        loss_kl = loss_kl_h + loss_kl_x

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        z_xh = self.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)
        diffusion_utils.assert_mean_zero_with_mask(z_xh[:, :, :self.n_dims], node_mask)

        # Decoder output (reconstruction).
        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        xh_rec = torch.cat([x_recon, h_recon], dim=2)
        loss_recon = self.compute_reconstruction_error(xh_rec, xh)

        # Combining the terms
        assert loss_recon.size() == loss_kl.size()
        loss = loss_recon + self.kl_weight * loss_kl

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'loss_t': loss.squeeze(), 'rec_error': loss_recon.squeeze()}

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the ELBO if training. And if eval then always computes NLL.
        """

        loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context)

        neg_log_pxh = loss

        return neg_log_pxh

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.latent_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def encode(self, x, h, node_mask=None, edge_mask=None, context=None):
        """Computes q(z|x)."""

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)

        diffusion_utils.assert_mean_zero_with_mask(xh[:, :, :self.n_dims], node_mask)

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encoder._forward(xh, node_mask, edge_mask, context)

        bs, _, _ = z_x_mu.size()
        sigma_0_x = torch.ones(bs, 1, 1).to(z_x_mu) * 0.0032
        sigma_0_h = torch.ones(bs, 1, self.latent_node_nf).to(z_h_mu) * 0.0032

        return z_x_mu, sigma_0_x, z_h_mu, sigma_0_h

    def decode(self, z_xh, node_mask=None, edge_mask=None, context=None):
        """Computes p(x|z)."""

        # Decoder output (reconstruction).

        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        diffusion_utils.assert_mean_zero_with_mask(x_recon, node_mask)

        xh = torch.cat([x_recon, h_recon], dim=2)

        x = xh[:, :, :self.n_dims]
        diffusion_utils.assert_correctly_masked(x, node_mask)

        h_int = xh[:, :, -1:] if self.include_charges else torch.zeros(0).to(xh)
        h_cat = xh[:, :, self.n_dims:-1]  # TODO: have issue when include_charges is False
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}

        return x, h

    @torch.no_grad()
    def reconstruct(self, x, h, node_mask=None, edge_mask=None, context=None):
        pass

    def log_info(self):
        """
        Some info logging of the model.
        """
        info = None
        print(info)

        return info


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class EnLatentDiffusion(EnVariationalDiffusion):
    """
    The E(n) Latent Diffusion Module.
    """

    def __init__(self, **kwargs):
        vae = kwargs.pop('vae')
        trainable_ae = kwargs.pop('trainable_ae', False)
        super().__init__(**kwargs)

        # Create self.vae as the first stage model.
        self.trainable_ae = trainable_ae
        self.instantiate_first_stage(vae)
        self.args = None

    def unnormalize_z(self, z, node_mask):
        # Overwrite the unnormalize_z function to do nothing (for sample_chain).

        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_int = z[:, :, self.n_dims + self.num_classes:self.n_dims + self.num_classes + 1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        # x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def log_constants_p_h_given_z0(self, h, node_mask):
        """Computes p(h|z0)."""
        batch_size = h.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_h = n_nodes * self.n_dims

        zeros = torch.zeros((h.size(0), 1), device=h.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_h * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)
        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        x = xh[:, :, :self.n_dims]

        # h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        # x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        # h_int = torch.round(h_int).long() * node_mask

        # Make the data structure compatible with the EnVariationalDiffusion sample() and sample_chain().
        h = {'integer': xh[:, :, self.n_dims:], 'categorical': torch.zeros(0).to(xh)}

        return x, h

    # ChemGuide
    def cond_sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False,
                                  cond_fn=None, guidance_kwargs=None, delta_z0=None):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        if delta_z0 is not None:
            alpha_t = self.alpha(gamma_0, target_tensor=net_out)
            # eq (9), https://arxiv.org/pdf/2302.07121
            net_out = net_out - alpha_t / torch.sqrt(1 - alpha_t * alpha_t) * delta_z0

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.cond_sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise,
                                     cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)

        x = xh[:, :, :self.n_dims]

        # h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        # x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)
        #
        # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        # h_int = torch.round(h_int).long() * node_mask

        # Make the data structure compatible with the EnVariationalDiffusion sample() and sample_chain().
        h = {'integer': xh[:, :, self.n_dims:], 'categorical': torch.zeros(0).to(xh)}
        return x, h

    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):

        # Computes the error for the distribution N(latent | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_pxh_given_z_without_constants = -0.5 * self.compute_error(net_out, gamma_0, eps)

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_pxh_given_z_without_constants

        return log_p_xh_given_z

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """

        # Encode data to latent space.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.vae.encode(x, h, node_mask, edge_mask, context)
        # Compute fixed sigma values.
        t_zeros = torch.zeros(size=(x.size(0), 1), device=x.device)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)  # make self.gamma(t_zeros) broadcastable to x
        sigma_0 = self.sigma(gamma_0, x)

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = sigma_0
        # z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        # TODO: I think we should use the line 1154 instead of line 1153 ?
        z_xh = self.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        z_xh = z_xh.detach()  # Always keep the encoder fixed.
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)

        # Compute reconstruction loss.
        if self.trainable_ae:
            xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
            # Decoder output (reconstruction).
            x_recon, h_recon = self.vae.decoder._forward(z_xh, node_mask, edge_mask, context)
            xh_rec = torch.cat([x_recon, h_recon], dim=2)
            loss_recon = self.vae.compute_reconstruction_error(xh_rec, xh)
        else:
            loss_recon = 0

        z_x = z_xh[:, :, :self.n_dims]  # 3d coordinates
        z_h = z_xh[:, :, self.n_dims:]  # node features
        diffusion_utils.assert_mean_zero_with_mask(z_x, node_mask)
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        z_h = {'categorical': torch.zeros(0).to(z_h), 'integer': z_h}

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss_ld, loss_dict = self.compute_loss(z_x, z_h, node_mask, edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss_ld, loss_dict = self.compute_loss(z_x, z_h, node_mask, edge_mask, context, t0_always=True)

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_h_given_z0(
            torch.cat([h['categorical'], h['integer']], dim=2), node_mask)
        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        neg_log_pxh = loss_ld + loss_recon + neg_log_constants

        return neg_log_pxh

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        """
        z_x, z_h = super().sample(n_samples, n_nodes, node_mask, edge_mask, context, fix_noise)

        z_xh = torch.cat([z_x, z_h['categorical'], z_h['integer']], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)
        x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)

        return x, h

    # ChemGuide
    @torch.no_grad()
    def cond_sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False,
                    cond_fn=None, use_neural_guidance=False, guidance_kwargs=None, save_traj_flag=False,
                    use_bilevel=None, args=None):
        """
        Draw (conditional) samples from the generative model with guidance.
        """
        self.args = args
        if cond_fn is None:
            if use_neural_guidance:
                cond_fn = self.cond_neural_fn
            else:
                if 'prop' in args.package:  # use 'prop' in the package's name to indicate
                    cond_fn = self.cond_fn_prop
                else:
                    cond_fn = self.cond_fn

        if use_bilevel:
            if self.args.bilevel_method == 'uni_guide':
                print('===============using universal guidance for property in bilevel optimization===============')
                z_x, z_h = super().cond_sample_bilevel_clean(n_samples, n_nodes, node_mask, edge_mask, context,
                                                             fix_noise, cond_force_fn=self.cond_fn,
                                                             cond_property_fn=self.cond_neural_fn_at_z0,
                                                             guidance_kwargs=guidance_kwargs,
                                                             save_traj_flag=save_traj_flag,
                                                             args=args)
            else:
                print('===============using regular RG guidance for bilevel optimization===============')
                z_x, z_h = super().cond_sample_bilevel_noisy(n_samples, n_nodes, node_mask, edge_mask, context,
                                                             fix_noise,
                                                             cond_force_fn=self.cond_fn,
                                                             cond_property_fn=self.cond_neural_fn,
                                                             guidance_kwargs=guidance_kwargs,
                                                             save_traj_flag=save_traj_flag,
                                                             args=args)
        elif self.args.use_evolution:
            z_x, z_h = super().cond_sample_evolutionary(n_samples, n_nodes, node_mask, edge_mask, context, fix_noise,
                                                        cond_fn=cond_fn,
                                                        guidance_kwargs=guidance_kwargs,
                                                        save_traj_flag=save_traj_flag,
                                                        args=args)
        else:  # single property (e.g., force, alpha, mu, etc.) guidance
            z_x, z_h = super().cond_sample(n_samples, n_nodes, node_mask, edge_mask, context, fix_noise,
                                           cond_fn=cond_fn,
                                           guidance_kwargs=guidance_kwargs,
                                           save_traj_flag=save_traj_flag,
                                           args=args)

        z_xh = torch.cat([z_x, z_h['categorical'], z_h['integer']], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)
        x0, h0 = self.vae.decode(z_xh, node_mask, edge_mask, context)

        if args.use_neural_guidance or args.use_cond_geo or args.bilevel_opti:  # has property regressor/classifier
            mae_loss = self.compute_MAE_w_NN(x0=x0, node_mask=node_mask, edge_mask=edge_mask, h0=h0,
                                             guidance_kwargs=guidance_kwargs)
            with open('{}/final_MAE_results.jsonl'.format(self.save_path), 'a+') as f:
                f.write(json.dumps(
                    {'run_id': args.run_id, 'clf_scale': args.clf_scale,
                     'clf_scale_force': args.clf_scale_force, 'clf_scale_prop': args.clf_scale_prop,
                     'batch_size': args.num_samples, 'MAE': mae_loss}) + '\n')
        if 'prop' in args.package:  # using xtb for property guidance
            mae_loss, z0_prop, z0_validity, z0_nan_cnt, z0_inf_cnt = self.compute_MAE_w_xtb(x0=x0, node_mask=node_mask,
                                                                                            edge_mask=edge_mask, h0=h0,
                                                                                            guidance_kwargs=guidance_kwargs)
            with open('{}/final_MAE_results.jsonl'.format(self.save_path), 'a+') as f:
                f.write(json.dumps(
                    {'run_id': args.run_id, 'clf_scale': args.clf_scale,
                     'clf_scale_force': args.clf_scale_force, 'clf_scale_prop': args.clf_scale_prop,
                     'batch_size': args.num_samples, 'MAE': mae_loss, 'z0_validity': z0_validity.sum().item(),
                     'z0_nan_cnt': z0_nan_cnt, 'z0_inf_cnt': z0_inf_cnt}) + '\n')

        return x0, h0

    # ChemGuide
    def compute_MAE_w_NN(self, x0, node_mask, edge_mask, h0, guidance_kwargs):
        print('in computing MAE')
        batch_size, n_nodes, _ = x0.size()
        atom_positions = x0.view(batch_size * n_nodes, -1).to(x0.device, torch.float32)
        atom_mask = node_mask.view(batch_size * n_nodes, -1).to(x0.device, torch.float32)
        edge_mask = edge_mask.to(x0.device, torch.float32)
        nodes = h0['categorical'].to(x0.device, torch.float32)

        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = prop_utils.get_adj_matrix(n_nodes, batch_size, x0.device)

        pred_y = self.args.cond_neural_model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None,
                                             node_mask=atom_mask,
                                             edge_mask=edge_mask,
                                             n_nodes=n_nodes)

        target_context = guidance_kwargs["target_context"]
        loss_l1 = nn.L1Loss()
        mean = self.args.mean
        mad = self.args.mad
        mae_loss = loss_l1(mad * pred_y + mean, mad * target_context + mean).item()
        return mae_loss

    # ChemGuide
    def compute_MAE_w_xtb(self, x0, node_mask, edge_mask, h0, guidance_kwargs):
        z0_prop, z0_validity, z0_nan_cnt, z0_inf_cnt = self.get_property_from_command_xtb(
            x=x0, h=h0, node_mask=node_mask, atom_mapping=guidance_kwargs["atom_mapping"],
            n_cores=guidance_kwargs["n_cores"], acc=guidance_kwargs["acc"],
            temp_dir=guidance_kwargs["temp_dir"], property=guidance_kwargs["property"]
        )

        loss_l1 = nn.L1Loss()
        mae_loss = loss_l1(z0_prop, guidance_kwargs["target_context"]).item()
        return mae_loss, z0_prop, z0_validity, z0_nan_cnt, z0_inf_cnt

    # ChemGuide for force
    def cond_fn(self, t, mu, zt, n_samples, node_mask, edge_mask, context, fix_noise, scale, target,
                atom_mapping, use_one_step, guidance_steps, grad_clip_threshold, n_cores, acc, temp_dir, get_force_fn,
                package, property, run_id, **kwargs):
        """
        Parameters
        ----------
        t: current timestep
        mu: latent variable at timestep t
        (the mean that should be passed into the regressor, as show in OpenAI's algorithm xt=mu)
        zt: latent variable at timestep t (zt)
        n_samples: for "sample_xh_given_zt"
        node_mask: for "sample_xh_given_zt"
        edge_mask: for "sample_xh_given_zt"
        context: for "sample_xh_given_zt"
        fix_noise: for "sample_xh_given_zt"
        scale: gradient scaling factor
        target: desired property
        atom_mapping: dictionary with key as idx (range over num_classes) and value as atomic number
        use_one_step: whether to use one/multiple-step diffusion to get z0
        guidance_steps: number of steps that we add the guidance
        grad_clip_threshold: clip the gradient if it's larger than the treshold
        Returns
        -------
        """
        if t <= guidance_steps and scale != 0:  # add guidance only in late diffusion steps (e.g., last 400 steps)
            if t % self.every_n_step == 0:
                epsilon = 1e-6
                gaussian_pert_tensor = utils.sample_center_gravity_zero_gaussian_with_mask(
                    size=zt.shape,
                    device=node_mask.device,
                    node_mask=node_mask)  # bs, n_node, feature_dim
                # mask the epsilon tensor
                gaussian_pert_tensor[:, :, self.n_dims:] = 0
                if use_one_step:
                    z0_x, z0_h = super().one_step_sample_xh_given_zt(n_samples=n_samples, zt=zt, timestep=t,
                                                                     node_mask=node_mask, edge_mask=edge_mask,
                                                                     context=context, fix_noise=fix_noise)
                else:
                    z0_x, z0_h = super().sample_xh_given_zt(n_samples=n_samples, z=mu, timestep=t, node_mask=node_mask,
                                                            edge_mask=edge_mask, context=context, fix_noise=fix_noise)
                z0_xh = torch.cat([z0_x, z0_h['categorical'], z0_h['integer']], dim=2)
                diffusion_utils.assert_correctly_masked(z0_xh, node_mask)

                x0, h0 = self.vae.decode(z0_xh, node_mask, edge_mask, context)

                z0_force, z0_validity, z0_nan_cnt, z0_inf_cnt = get_force_fn(x=x0, h=h0, node_mask=node_mask,
                                                                             atom_mapping=atom_mapping,
                                                                             n_cores=n_cores,
                                                                             acc=acc,
                                                                             temp_dir=temp_dir)  # (bs, n, 3), (bs, )
                # compute F(zt+epsilon) to evaluate grad_{zt}[y_hat]
                if use_one_step:
                    z0_plus_x, z0_plus_h = super().one_step_sample_xh_given_zt(n_samples=n_samples,
                                                                               zt=zt + epsilon * gaussian_pert_tensor,
                                                                               timestep=t,
                                                                               node_mask=node_mask, edge_mask=edge_mask,
                                                                               context=context, fix_noise=fix_noise)
                else:
                    z0_plus_x, z0_plus_h = super().sample_xh_given_zt(n_samples=n_samples,
                                                                      z=mu + epsilon * gaussian_pert_tensor, timestep=t,
                                                                      node_mask=node_mask,
                                                                      edge_mask=edge_mask, context=context,
                                                                      fix_noise=fix_noise)
                z0_plus_xh = torch.cat([z0_plus_x, z0_plus_h['categorical'], z0_plus_h['integer']], dim=2)
                diffusion_utils.assert_correctly_masked(z0_plus_xh, node_mask)
                x0_plus, h0_plus = self.vae.decode(z0_plus_xh, node_mask, edge_mask, context)

                z0_plus_force, z0_plus_validity, z0_plus_nan_cnt, z0_plus_inf_cnt = get_force_fn(x=x0_plus, h=h0_plus,
                                                                                                 node_mask=node_mask,
                                                                                                 atom_mapping=atom_mapping,
                                                                                                 n_cores=n_cores,
                                                                                                 acc=acc,
                                                                                                 temp_dir=temp_dir)
                # compute F(zt-epsilon) to evaluate grad_{zt}[y_hat]
                if use_one_step:
                    z0_minus_x, z0_minus_h = super().one_step_sample_xh_given_zt(n_samples=n_samples,
                                                                                 zt=zt - epsilon * gaussian_pert_tensor,
                                                                                 timestep=t,
                                                                                 node_mask=node_mask,
                                                                                 edge_mask=edge_mask,
                                                                                 context=context, fix_noise=fix_noise)
                else:
                    z0_minus_x, z0_minus_h = super().sample_xh_given_zt(n_samples=n_samples,
                                                                        z=mu - epsilon * gaussian_pert_tensor,
                                                                        timestep=t,
                                                                        node_mask=node_mask,
                                                                        edge_mask=edge_mask, context=context,
                                                                        fix_noise=fix_noise)

                z0_minus_xh = torch.cat([z0_minus_x, z0_minus_h['categorical'], z0_minus_h['integer']], dim=2)
                diffusion_utils.assert_correctly_masked(z0_minus_xh, node_mask)
                x0_minus, h0_minus = self.vae.decode(z0_minus_xh, node_mask, edge_mask, context)

                z0_minus_force, z0_minus_validity, z0_minus_nan_cnt, z0_minus_inf_cnt = get_force_fn(x=x0_minus,
                                                                                                     h=h0_minus,
                                                                                                     node_mask=node_mask,
                                                                                                     atom_mapping=atom_mapping,
                                                                                                     n_cores=n_cores,
                                                                                                     acc=acc,
                                                                                                     temp_dir=temp_dir)

                with open('{}/record_nan_in_xtb.jsonl'.format(self.save_path), 'a+') as f:
                    f.write(json.dumps(
                        {'t': t, 'z0_nan': z0_nan_cnt, 'z0_inf': z0_inf_cnt, 'z0_valid': z0_validity.sum().item(),
                         'z0_plus_nan': z0_plus_nan_cnt, 'z0_plus_inf': z0_plus_inf_cnt,
                         'z0_plus_valid': z0_plus_validity.sum().item(),
                         'z0_minus_nan': z0_minus_nan_cnt, 'z0_minus_inf': z0_minus_inf_cnt,
                         'z0_minus_valid': z0_minus_validity.sum().item(), "run_id": run_id
                         }) + '\n')

                # approximate grad_{zt}[y_hat] = grad_{zt}[F(zt)]

                left_mask = z0_validity * z0_minus_validity  # (bs, )
                right_mask = z0_validity * z0_plus_validity  # (bs, )
                coef = 0.5
                left_mask = coef * left_mask  # convex combination of the subgradients
                right_mask = (1 - coef) * right_mask  # convex combination of the subgradients
                left_mask[right_mask == 0] = 1  # use only left gradient (weight set to 1)
                # and convert no left-right grads into left grad only
                right_mask[left_mask == 0] = 1  # use only right gradient (weight set to 1)

                # convex combination of the subgradients
                left_grad = left_mask.view(-1, 1, 1) * (z0_minus_force - z0_force) / -epsilon
                right_grad = right_mask.view(-1, 1, 1) * (z0_plus_force - z0_force) / epsilon
                if torch.isnan(left_grad).any():
                    print('existing nan in left grad')
                if torch.isnan(right_grad).any():
                    print('existing nan in right grad')

                approx_grad = left_grad + right_grad
                # (bs, n, 3)
                approx_grad = approx_grad.unsqueeze(dim=-1).expand(-1, -1, -1,
                                                                   zt.size(-1)) * gaussian_pert_tensor.unsqueeze(2)
                # (bs, n, 3) ---> (bs, n, 3, z_dim)
                if torch.isnan(approx_grad).any():
                    print('existing nan in approx grad')

                validity_mask = z0_minus_validity + z0_plus_validity
                validity_mask[validity_mask != 0] = 1
                validity_mask = validity_mask * z0_validity
                grad = ((z0_force - target).unsqueeze(dim=-2) @ approx_grad).squeeze(dim=-2) * \
                       validity_mask.view(-1, 1, 1)
                # (bs, n, 1, 3) @ (bs, n, 3, z_dim) ---> (bs, n, 1, z_dim) ---> (bs, n, z_dim)
                if torch.isnan(grad).any():
                    print('existing nan in grad before clip')

                if grad_clip_threshold != float("inf"):
                    grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True).expand(-1, -1, grad.size(-1))
                    grad_mask = (grad_norm > grad_clip_threshold)
                    grad[grad_mask] = grad[grad_mask] / grad_norm[grad_mask] * grad_clip_threshold
                    grad[torch.isnan(grad)] = 0

                guidance_grad = -scale * grad * node_mask

                self.hist_grad_force = guidance_grad
            else:  # for the rest (n-1) steps
                if self.use_hist_grad:  # use the previously stored guidance
                    guidance_grad = self.hist_grad_force
                else:  # no guidance
                    guidance_grad = 0
            return guidance_grad

        else:  # do not add guidance until the last "guidance_steps"
            return 0

    # ChemGuide for property
    def cond_fn_prop(self, t, mu, zt, n_samples, node_mask, edge_mask, context, fix_noise, scale, target_context,
                     atom_mapping, use_one_step, guidance_steps, grad_clip_threshold, n_cores, acc, temp_dir,
                     get_force_fn, package, property, run_id, **kwargs):
        """
        Parameters
        ----------
        t: current timestep
        mu: latent variable at timestep t
        (the mean that should be passed into the regressor, as show in OpenAI's algorithm xt=mu)
        zt: latent variable at timestep t (zt)
        n_samples: for "sample_xh_given_zt"
        node_mask: for "sample_xh_given_zt"
        edge_mask: for "sample_xh_given_zt"
        context: for "sample_xh_given_zt"
        fix_noise: for "sample_xh_given_zt"
        scale: gradient scaling factor
        target: desired property
        atom_mapping: dictionary with key as idx (range over num_classes) and value as atomic number
        use_one_step: whether to use one/multiple-step diffusion to get z0
        guidance_steps: number of steps that we add the guidance
        grad_clip_threshold: clip the gradient if it's larger than the treshold
        Returns
        -------
        """
        if t <= guidance_steps and scale != 0:  # add guidance only in late diffusion steps (e.g., last 200 steps)
            if t % self.every_n_step == 0:
                mean = self.args.mean
                mad = self.args.mad

                epsilon = 1e-6
                gaussian_pert_tensor = utils.sample_center_gravity_zero_gaussian_with_mask(
                    size=zt.shape,
                    device=node_mask.device,
                    node_mask=node_mask)  # bs, n_node, feature_dim

                # z0 is used to get y_hat from xtb
                if use_one_step:
                    z0_x, z0_h = super().one_step_sample_xh_given_zt(n_samples=n_samples, zt=zt, timestep=t,
                                                                     node_mask=node_mask, edge_mask=edge_mask,
                                                                     context=context, fix_noise=fix_noise)
                else:
                    z0_x, z0_h = super().sample_xh_given_zt(n_samples=n_samples, z=mu, timestep=t, node_mask=node_mask,
                                                            edge_mask=edge_mask, context=context, fix_noise=fix_noise)
                z0_xh = torch.cat([z0_x, z0_h['categorical'], z0_h['integer']], dim=2)
                diffusion_utils.assert_correctly_masked(z0_xh, node_mask)

                x0, h0 = self.vae.decode(z0_xh, node_mask, edge_mask, context)

                z0_prop, z0_validity, z0_nan_cnt, z0_inf_cnt = get_force_fn(x=x0, h=h0, node_mask=node_mask,
                                                                            atom_mapping=atom_mapping,
                                                                            n_cores=n_cores,
                                                                            acc=acc,
                                                                            temp_dir=temp_dir,
                                                                            property=property)  # (bs, ), (bs, )
                # compute F(zt+epsilon) to evaluate grad_{zt}[y_hat]
                if use_one_step:
                    z0_plus_x, z0_plus_h = super().one_step_sample_xh_given_zt(n_samples=n_samples,
                                                                               zt=zt + epsilon * gaussian_pert_tensor,
                                                                               timestep=t,
                                                                               node_mask=node_mask, edge_mask=edge_mask,
                                                                               context=context, fix_noise=fix_noise)
                else:
                    z0_plus_x, z0_plus_h = super().sample_xh_given_zt(n_samples=n_samples,
                                                                      z=mu + epsilon * gaussian_pert_tensor, timestep=t,
                                                                      node_mask=node_mask,
                                                                      edge_mask=edge_mask, context=context,
                                                                      fix_noise=fix_noise)
                z0_plus_xh = torch.cat([z0_plus_x, z0_plus_h['categorical'], z0_plus_h['integer']], dim=2)
                diffusion_utils.assert_correctly_masked(z0_plus_xh, node_mask)
                x0_plus, h0_plus = self.vae.decode(z0_plus_xh, node_mask, edge_mask, context)

                z0_plus_prop, z0_plus_validity, z0_plus_nan_cnt, z0_plus_inf_cnt = get_force_fn(x=x0_plus, h=h0_plus,
                                                                                                node_mask=node_mask,
                                                                                                atom_mapping=atom_mapping,
                                                                                                n_cores=n_cores,
                                                                                                acc=acc,
                                                                                                temp_dir=temp_dir,
                                                                                                property=property)
                # compute F(zt-epsilon) to evaluate grad_{zt}[y_hat]
                if use_one_step:
                    z0_minus_x, z0_minus_h = super().one_step_sample_xh_given_zt(n_samples=n_samples,
                                                                                 zt=zt - epsilon * gaussian_pert_tensor,
                                                                                 timestep=t,
                                                                                 node_mask=node_mask,
                                                                                 edge_mask=edge_mask,
                                                                                 context=context, fix_noise=fix_noise)
                else:
                    z0_minus_x, z0_minus_h = super().sample_xh_given_zt(n_samples=n_samples,
                                                                        z=mu - epsilon * gaussian_pert_tensor,
                                                                        timestep=t,
                                                                        node_mask=node_mask,
                                                                        edge_mask=edge_mask, context=context,
                                                                        fix_noise=fix_noise)

                z0_minus_xh = torch.cat([z0_minus_x, z0_minus_h['categorical'], z0_minus_h['integer']], dim=2)
                diffusion_utils.assert_correctly_masked(z0_minus_xh, node_mask)
                x0_minus, h0_minus = self.vae.decode(z0_minus_xh, node_mask, edge_mask, context)
                # (B, max_node, num_classes)
                z0_minus_prop, z0_minus_validity, z0_minus_nan_cnt, z0_minus_inf_cnt = get_force_fn(x=x0_minus,
                                                                                                    h=h0_minus,
                                                                                                    node_mask=node_mask,
                                                                                                    atom_mapping=atom_mapping,
                                                                                                    n_cores=n_cores,
                                                                                                    acc=acc,
                                                                                                    temp_dir=temp_dir,
                                                                                                    property=property)
                # (bs, ), (bs, )
                with open('{}/record_nan_in_xtb.jsonl'.format(self.save_path), 'a+') as f:
                    loss_l1 = nn.L1Loss()
                    loss_l2 = nn.MSELoss()
                    # loss = loss_l1(mad * pred + mean, label) ---> label = context * mad + mean
                    mae_loss = loss_l1(z0_prop, target_context).item()
                    mse_loss = loss_l2(z0_prop, target_context).item()
                    print(
                        f'z0 prop vs. target context: mae-loss={mae_loss},  mse-loss={mse_loss}, mad={mad}, mean={mean}')
                    f.write(json.dumps(
                        {'t': t, 'z0_nan': z0_nan_cnt, 'z0_inf': z0_inf_cnt, 'z0_valid': z0_validity.sum().item(),
                         'z0_plus_nan': z0_plus_nan_cnt, 'z0_plus_inf': z0_plus_inf_cnt,
                         'z0_plus_valid': z0_plus_validity.sum().item(),
                         'z0_minus_nan': z0_minus_nan_cnt, 'z0_minus_inf': z0_minus_inf_cnt,
                         'z0_minus_valid': z0_minus_validity.sum().item(), "run_id": run_id,
                         'MAE': mae_loss, 'MSE': mse_loss
                         }) + '\n')

                # approximate grad_{zt}[y_hat] = grad_{zt}[F(zt)]
                # (B, max_node, num_classes=3)
                left_mask = z0_validity * z0_minus_validity  # (bs, )
                right_mask = z0_validity * z0_plus_validity  # (bs, )
                coef = 0.5
                left_mask = coef * left_mask  # convex combination of the subgradients
                right_mask = (1 - coef) * right_mask  # convex combination of the subgradients
                left_mask[right_mask == 0] = 1  # use only left gradient (weight set to 1)
                # and convert no left-right grads into left grad only
                right_mask[left_mask == 0] = 1  # use only right gradient (weight set to 1)

                # convex combination of the subgradients
                left_grad = left_mask * (z0_minus_prop - z0_prop) / -epsilon
                right_grad = right_mask * (z0_plus_prop - z0_prop) / epsilon
                if torch.isnan(left_grad).any():
                    print('existing nan in left grad')
                if torch.isnan(right_grad).any():
                    print('existing nan in right grad')

                approx_grad = left_grad + right_grad  # (bs, )
                approx_grad = approx_grad.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1, -1, zt.size(-1)) * \
                              gaussian_pert_tensor
                # (bs, 1, 1) * (bs, n, z_dim) ---> (bs, n, z_dim)
                if torch.isnan(approx_grad).any():
                    print('existing nan in approx grad')

                validity_mask = z0_minus_validity + z0_plus_validity
                validity_mask[validity_mask != 0] = 1
                validity_mask = validity_mask * z0_validity
                grad = ((z0_prop - target_context).unsqueeze(dim=-1).unsqueeze(dim=-1) * approx_grad) * \
                       validity_mask.view(-1, 1, 1)
                # (bs, 1, 1)[L2 here only acts as scaling] * (bs, n, z_dim) * (bs, 1, 1)[validity mask]
                # ---> (bs, n, z_dim)
                if torch.isnan(grad).any():
                    print('existing nan in grad before clip')

                if grad_clip_threshold != float("inf"):
                    grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True).expand(-1, -1, grad.size(-1))
                    grad_mask = (grad_norm > grad_clip_threshold)
                    grad[grad_mask] = grad[grad_mask] / grad_norm[grad_mask] * grad_clip_threshold
                    grad[torch.isnan(grad)] = 0

                guidance_grad = -scale * grad * node_mask

                self.hist_grad_force = guidance_grad
            else:  # for the rest (n-1) steps
                if self.use_hist_grad:  # use the previously stored guidance
                    guidance_grad = self.hist_grad_force
                else:  # no guidance
                    guidance_grad = 0
            return guidance_grad

        else:  # do not add guidance until the last "guidance_steps"
            return 0

    # ChemGuide for property regressor noisy guidance
    def cond_neural_fn(self, t, mu, zt, n_samples, node_mask, edge_mask, context, fix_noise, scale, target,
                       atom_mapping, use_one_step, guidance_steps, grad_clip_threshold, cond_neural_model,
                       target_context, **kwargs):
        """
        Parameters
        ----------
        t: current timestep
        mu: latent variable at timestep t
        (the mean that should be passed into the regressor, as show in OpenAI's algorithm xt=mu)
        zt: latent variable at timestep t (zt)
        n_samples: for "sample_xh_given_zt"
        node_mask: for "sample_xh_given_zt"
        edge_mask: for "sample_xh_given_zt"
        context: for "sample_xh_given_zt"
        fix_noise: for "sample_xh_given_zt"
        scale: gradient scaling factor
        target: desired property
        atom_mapping: dictionary with key as idx (range over num_classes) and value as atomic number
        use_one_step: whether to use one/multiple-step diffusion to get z0
        guidance_steps: number of steps that we add the guidance
        grad_clip_threshold: clip the gradient if it's larger than the treshold
        Returns
        -------
        """
        if t <= guidance_steps and scale != 0:  # add guidance only in late diffusion steps (e.g., last 200 steps)
            loss = nn.MSELoss()
            if t % self.every_n_step == 0:
                with torch.enable_grad():
                    zt_in = zt.detach().requires_grad_(True)
                    # here we can only use the one_step_sample method
                    z0_x, z0_h = super().one_step_sample_xh_given_zt(n_samples=n_samples, zt=zt_in, timestep=t,
                                                                     node_mask=node_mask, edge_mask=edge_mask,
                                                                     context=context, fix_noise=fix_noise)

                    z0_xh = torch.cat([z0_x, z0_h['categorical'], z0_h['integer']], dim=2)
                    diffusion_utils.assert_correctly_masked(z0_xh, node_mask)
                    x0, h0 = self.vae.decode(z0_xh, node_mask, edge_mask, context)

                    batch_size, n_nodes, _ = zt_in.size()
                    atom_positions = x0.view(batch_size * n_nodes, -1).to(zt_in.device, torch.float32)
                    atom_mask = node_mask.view(batch_size * n_nodes, -1).to(zt_in.device, torch.float32)
                    edge_mask = edge_mask.to(zt_in.device, torch.float32)
                    nodes = h0['categorical'].to(zt_in.device, torch.float32)

                    nodes = nodes.view(batch_size * n_nodes, -1)
                    edges = prop_utils.get_adj_matrix(n_nodes, batch_size, zt_in.device)

                    pred_y = cond_neural_model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None,
                                               node_mask=atom_mask,
                                               edge_mask=edge_mask,
                                               n_nodes=n_nodes)
                    mean = self.args.mean
                    mad = self.args.mad

                    # (bs, 1) or (bs, )
                    mse = loss(mad * pred_y + mean, mad * target_context + mean)  # bs, 1

                    with open('{}/record_mse_mae_in_neural_guidance.jsonl'.format(self.save_path), 'a+') as f:
                        loss_l1 = nn.L1Loss()
                        mae_loss = loss_l1(mad * pred_y + mean, mad * target_context + mean).item()
                        f.write(json.dumps(
                            {'t': t, 'mse_loss': mse.item(), 'run_id': self.args.run_id,
                             'scale': scale, 'batch_size': self.args.num_samples, 'MAE': mae_loss}) + '\n')
                        if t <= 50:  # begin to record the last MAE and decide whether to add guidance
                            if self.last_property_MAE is None:
                                self.last_property_MAE = mae_loss
                            else:
                                if (mae_loss // self.last_property_MAE) >= 5:
                                    return 0  # no property guidance
                                else:
                                    self.last_property_MAE = mae_loss  # update the latest recorded MAE
                    if t % 100 == 0 and self.args.bilevel_opti:
                        with open(f'{self.save_path}/{t}_decoded_xh0.pickle', 'wb') as handle:
                            inst = {'x0': x0.to('cpu'), 'h0': {'integer': h0['integer'].to('cpu'),
                                                               'categorical': h0['categorical'].to('cpu')}}
                            pickle.dump(inst, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    grad = torch.autograd.grad(mse, zt_in)[0]  # same shape as zt_in: (bs, n_node, feature_dim[3+2])

                    if grad_clip_threshold != float("inf"):
                        grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True).expand(-1, -1, grad.size(-1))
                        grad_mask = (grad_norm > grad_clip_threshold)
                        grad[grad_mask] = grad[grad_mask] / grad_norm[grad_mask] * grad_clip_threshold
                        grad[torch.isnan(grad)] = 0

                    guidance_grad = -scale * grad * node_mask

                    self.hist_grad_property = guidance_grad
            else:  # for the rest (n-1) steps
                if self.use_hist_grad:  # use the previously stored guidance
                    guidance_grad = self.hist_grad_property
                else:  # no guidance
                    guidance_grad = 0
            return guidance_grad

        else:  # do not add guidance until the last "guidance_steps"
            return 0

    # ChemGuide for property regressor clean guidance
    def cond_neural_fn_at_z0(self, t, step, mu, z0, delta_z0, n_samples, node_mask, edge_mask, context, fix_noise,
                             scale, target, atom_mapping, use_one_step, guidance_steps, grad_clip_threshold,
                             cond_neural_model, target_context, **kwargs):
        """
        Parameters
        ----------
        t: current timestep
        mu: latent variable at timestep t
        (the mean that should be passed into the regressor, as show in OpenAI's algorithm xt=mu)
        zt: latent variable at timestep t (zt)
        n_samples: for "sample_xh_given_zt"
        node_mask: for "sample_xh_given_zt"
        edge_mask: for "sample_xh_given_zt"
        context: for "sample_xh_given_zt"
        fix_noise: for "sample_xh_given_zt"
        scale: gradient scaling factor
        target: desired property
        atom_mapping: dictionary with key as idx (range over num_classes) and value as atomic number
        use_one_step: whether to use one/multiple-step diffusion to get z0
        guidance_steps: number of steps that we add the guidance
        grad_clip_threshold: clip the gradient if it's larger than the threshold
        Returns
        -------
        """
        if t <= guidance_steps and scale != 0:  # add guidance only in late diffusion steps (e.g., last 200 steps)
            loss = nn.MSELoss()
            print('================property scale{}================'.format(scale))
            with torch.enable_grad():
                delta_z0_in = delta_z0.detach().requires_grad_(True)
                z0_xh = z0 + delta_z0_in
                x0, h0 = self.vae.decode(z0_xh, node_mask, edge_mask, context)

                batch_size, n_nodes, _ = z0.size()
                atom_positions = x0.view(batch_size * n_nodes, -1).to(z0.device, torch.float32)
                atom_mask = node_mask.view(batch_size * n_nodes, -1).to(z0.device, torch.float32)
                edge_mask = edge_mask.to(z0.device, torch.float32)
                nodes = h0['categorical'].to(z0.device, torch.float32)

                nodes = nodes.view(batch_size * n_nodes, -1)
                edges = prop_utils.get_adj_matrix(n_nodes, batch_size, z0.device)

                pred_y = cond_neural_model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None,
                                           node_mask=atom_mask,
                                           edge_mask=edge_mask,
                                           n_nodes=n_nodes)
                mean = self.args.mean
                mad = self.args.mad

                mse = loss(mad * pred_y + mean, mad * target_context + mean)  # bs, 1

                with open('{}/record_mse_mae_in_neural_guidance.jsonl'.format(self.save_path), 'a+') as f:
                    loss_l1 = nn.L1Loss()
                    mae_loss = loss_l1(mad * pred_y + mean, mad * target_context + mean).item()
                    f.write(json.dumps(
                        {'t': t, 'step': step, 'mse_loss': mse.item(), 'run_id': self.args.run_id,
                         'scale': scale, 'batch_size': self.args.num_samples, 'MAE': mae_loss}) + '\n')

                if t % 100 == 0 and self.args.bilevel_opti:
                    with open(f'{self.save_path}/{t}_step{step}_decoded_xh0.pickle', 'wb') as handle:
                        inst = {'x0': x0.to('cpu'), 'h0': {'integer': h0['integer'].to('cpu'),
                                                           'categorical': h0['categorical'].to('cpu')}}
                        pickle.dump(inst, handle, protocol=pickle.HIGHEST_PROTOCOL)

                grad = torch.autograd.grad(mse, delta_z0_in)[0]  # same shape as zt_in: (bs, n_node, feature_dim[3+2])

                if grad_clip_threshold != float("inf"):
                    grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True).expand(-1, -1, grad.size(-1))
                    grad_mask = (grad_norm > grad_clip_threshold)
                    grad[grad_mask] = grad[grad_mask] / grad_norm[grad_mask] * grad_clip_threshold
                    grad[torch.isnan(grad)] = 0

                guidance_grad = -scale * grad * node_mask

                return guidance_grad

        else:  # do not add guidance until the last "guidance_steps"
            return 0

    # ChemGuide
    def get_force_from_command_xtb(self, x: torch.Tensor, h: dict, node_mask: torch.Tensor,
                                   atom_mapping: dict, **kwargs):
        n_cores, acc, temp_dir = kwargs["n_cores"], kwargs["acc"], kwargs["temp_dir"]
        property_name = kwargs.get("property", None)
        if property_name is None:
            property_name = "force"
        here = os.getcwd()
        if temp_dir in here:
            os.chdir("..")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
        os.chdir(temp_dir)
        h_atom_types = torch.argmax(h['categorical'], dim=-1).cpu()  # (bs, n, num_classes) ---> (bs, n)
        h_atom_types.apply_(atom_mapping.get)

        # https://discuss.pytorch.org/t/map-the-value-in-a-tensor-using-dictionary/100933
        atom_forces = torch.zeros_like(x, device=x.device)  # (bs, n, 3)
        valid_atoms = []  # if xtb raises an error [the decoded atom from VAE is invalid],
        # we put a zero-mask on that molecule, otherwise, we put a one-mask
        nan_cnt = 0
        inf_cnt = 0
        # prepare input for multiprocessing
        pool_input = []
        for i, (positions, atoms, mask) in enumerate(zip(x, h_atom_types, node_mask.squeeze(-1))):
            # position: (n, 3)
            # atoms: (n, )
            # mask: (n, )
            mask = (mask == 1).cpu()  # transform mask into bool type tensor
            numbers = atoms.cpu()[mask].numpy()
            positions = positions.cpu()[mask, :].numpy()
            pool_input.append((positions, numbers, acc, i, property_name))
        with multiprocessing.Pool(self.args.n_cpus) as pool:
            pool_output = list(pool.starmap(run_xtb_command, pool_input))
            pool_output = sorted(pool_output, key=lambda _x: _x[1])

        for i, (positions, atoms, mask, force) in enumerate(zip(x, h_atom_types, node_mask.squeeze(-1), pool_output)):
            mask = (mask == 1).cpu()  # transform mask into bool type tensor
            numbers = atoms.cpu()[mask].numpy()
            positions = positions.cpu()[mask, :].numpy()
            n_node = positions.shape[0]
            force = force[0]
            if force is None:
                force = torch.zeros(n_node, 3)
                valid_atoms.append(0)
            else:
                valid_atoms.append(1)
                inf_flag = torch.isinf(force)
                nan_flag = torch.isnan(force)
                if torch.isinf(force).any():
                    force[inf_flag] = 0
                    inf_cnt += 1
                if nan_flag.any():
                    force[nan_flag] = 0
                    nan_cnt += 1
            # https://xtb-python.readthedocs.io/en/latest/general-api.html#single-point-calculator
            force = force.to(x.device)
            atom_forces[i, :n_node, :] = force
        if torch.isnan(atom_forces).any():
            print('inside command line, detected nan in atom forces')
        valid_atoms = torch.tensor(valid_atoms, device=x.device)
        print(f"max force {atom_forces.max().item()} "
              f"min force {atom_forces.min().item()} "
              f"avg force {atom_forces.mean().item()}")
        os.chdir(here)
        shutil.rmtree(temp_dir)
        return atom_forces, valid_atoms, nan_cnt, inf_cnt

    # ChemGuide
    def get_property_from_command_xtb(self, x: torch.Tensor, h: dict, node_mask: torch.Tensor,
                                      atom_mapping: dict, **kwargs):
        n_cores, acc, temp_dir = kwargs["n_cores"], kwargs["acc"], kwargs["temp_dir"]
        property_name = kwargs.get("property", None)
        here = os.getcwd()
        if temp_dir in here:
            os.chdir("..")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
        os.chdir(temp_dir)
        h_atom_types = torch.argmax(h['categorical'], dim=-1).cpu()  # (bs, n, num_classes) ---> (bs, n)
        h_atom_types.apply_(atom_mapping.get)

        # https://discuss.pytorch.org/t/map-the-value-in-a-tensor-using-dictionary/100933
        atom_property = torch.zeros(x.shape[0], device=x.device)  # (bs, n, 3)
        valid_atoms = []  # if xtb raises an error [the decoded atom from VAE is invalid],
        # we put a zero-mask on that molecule, otherwise, we put a one-mask
        nan_cnt = 0
        inf_cnt = 0
        # prepare input for multiprocessing
        pool_input = []
        for i, (positions, atoms, mask) in enumerate(zip(x, h_atom_types, node_mask.squeeze(-1))):
            # position: (n, 3)
            # atoms: (n, )
            # mask: (n, )
            mask = (mask == 1).cpu()  # transform mask into bool type tensor
            numbers = atoms.cpu()[mask].numpy()
            positions = positions.cpu()[mask, :].numpy()
            pool_input.append((positions, numbers, acc, i, property_name))
        with multiprocessing.Pool(self.args.n_cpus) as pool:
            pool_output = list(pool.starmap(run_xtb_command, pool_input))
            pool_output = sorted(pool_output, key=lambda _x: _x[1])

        for i, (property_score) in enumerate(pool_output):
            property_score = property_score[0]
            if property_score is None:
                property_score = torch.tensor(0.)
                valid_atoms.append(0.)
            else:
                valid_atoms.append(1)
                inf_flag = torch.isinf(property_score)
                nan_flag = torch.isnan(property_score)
                if torch.isinf(property_score).any():
                    property_score[inf_flag] = 0
                    inf_cnt += 1
                if nan_flag.any():
                    property_score[nan_flag] = 0
                    nan_cnt += 1
            # https://xtb-python.readthedocs.io/en/latest/general-api.html#single-point-calculator
            property_score = property_score.to(x.device)
            atom_property[i] = property_score
        if torch.isnan(atom_property).any():
            print('inside command line, detected nan in atom property')
        valid_atoms = torch.tensor(valid_atoms, device=x.device)
        print(f"max property {atom_property.max().item()} "
              f"min property {atom_property.min().item()} "
              f"avg property {atom_property.mean().item()}")

        os.chdir(here)
        shutil.rmtree(temp_dir)
        return atom_property, valid_atoms, nan_cnt, inf_cnt

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        chain_flat = super().sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames)

        # xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # chain[0] = xh  # Overwrite last frame with the resulting x and h.

        # chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        chain = chain_flat.view(keep_frames, n_samples, *chain_flat.size()[1:])
        chain_decoded = torch.zeros(
            size=(*chain.size()[:-1], self.vae.in_node_nf + self.vae.n_dims), device=chain.device)

        for i in range(keep_frames):
            z_xh = chain[i]
            diffusion_utils.assert_mean_zero_with_mask(z_xh[:, :, :self.n_dims], node_mask)

            x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)
            xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
            chain_decoded[i] = xh

        chain_decoded_flat = chain_decoded.view(n_samples * keep_frames, *chain_decoded.size()[2:])

        return chain_decoded_flat

    def select_best_evolution(self, zt_list, n_samples, t, node_mask, edge_mask, context, fix_noise, get_force_fn,
                              atom_mapping, n_cores, acc, temp_dir, **kwargs) -> list:
        best_batch = None
        best_force = float("inf")
        print("evolving...")
        for zt in zt_list:
            z0_x, z0_h = super().one_step_sample_xh_given_zt(n_samples=n_samples, zt=zt, timestep=t,
                                                             node_mask=node_mask, edge_mask=edge_mask,
                                                             context=context, fix_noise=fix_noise)

            z0_xh = torch.cat([z0_x, z0_h['categorical'], z0_h['integer']], dim=2)
            diffusion_utils.assert_correctly_masked(z0_xh, node_mask)

            x0, h0 = self.vae.decode(z0_xh, node_mask, edge_mask, context)

            z0_force, z0_validity, z0_nan_cnt, z0_inf_cnt = get_force_fn(x=x0, h=h0, node_mask=node_mask,
                                                                         atom_mapping=atom_mapping,
                                                                         n_cores=n_cores,
                                                                         acc=acc,
                                                                         temp_dir=temp_dir)  # (bs, n, 3), (bs, )

            current_force_rms = torch.sqrt(torch.square(z0_force).sum(dim=(1, 2)) / torch.sum(node_mask, dim=(1, 2)))
            current_force_rms = (current_force_rms * z0_validity).mean()

            if best_force > current_force_rms and z0_validity.sum() > 0:
                best_batch = zt
                best_force = current_force_rms
        if best_batch is None:
            best_batch = zt_list[0]
        return [best_batch]

    def instantiate_first_stage(self, vae: EnHierarchicalVAE):
        if not self.trainable_ae:
            self.vae = vae.eval()
            self.vae.train = disabled_train
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae = vae.train()
            for param in self.vae.parameters():
                param.requires_grad = True


def gradient_from(file_identifier=""):
    raw = []
    grad_filename = f"{file_identifier}.gradient" if file_identifier != "" else "gradient"
    if os.path.exists(grad_filename):
        with open(grad_filename, "r") as grad_file:
            for i, line in enumerate(grad_file):
                if i > 1 and len(line.split()) == 3:
                    x, y, z = line.split()
                    vec = [
                        float(x.replace("D", "E")),
                        float(y.replace("D", "E")),
                        float(z.replace("D", "E")),
                    ]
                    raw.append(vec)
    if len(raw) == 0:
        raise ValueError("Cannot get gradient")
    return raw


def property_from(property_name, file_identifier=""):
    """
    property_name: {energy, alpha, mu, Cv, homo, lumo, gap}
    """
    property_dict = {"energy": None, "alpha": None, "mu": None, "Cv": None, "homo": None, "lumo": None, "gap": None}
    if os.path.exists(f"mol_output_{file_identifier}.txt"):

        with open(f"mol_output_{file_identifier}.txt", "r") as f:
            # WARNING: the order of the following blocks is important
            # WARNING: DO NOT CHANGE THE ORDER OF THE FOLLOWING `IF` BLOCKS
            if property_name in ["gap", "lumo", "homo"]:
                while line := f.readline():
                    line = line.strip()
                    if "convergence criteria satisfied" in line:
                        break
                while line := f.readline():
                    line = line.strip()
                    if line.endswith("(HOMO)"):
                        try:
                            property_dict["homo"] = float(line.split()[3])
                        except:
                            pass
                    elif line.endswith("(LUMO)"):
                        try:
                            property_dict["lumo"] = float(line.split()[2])
                        except:
                            pass
                    elif line.startswith("HL-Gap"):
                        try:
                            property_dict["gap"] = float(line.split()[3])
                        finally:
                            break
            if property_name == "alpha":
                while line := f.readline():
                    line = line.strip()
                    if line.startswith("Mol. α(0) /au"):
                        try:
                            property_dict["alpha"] = float(line.split()[-1])
                        finally:
                            break
            if property_name == "mu":
                while line := f.readline():
                    line = line.strip()
                    if line.startswith("molecular dipole:"):
                        try:
                            f.readline()
                            f.readline()
                            line = f.readline().strip()
                            property_dict["mu"] = float(line.split()[-1])
                        finally:
                            break
            if property_name == "Cv":
                while line := f.readline():
                    line = line.strip()
                    if line.endswith("enthalpy   heat capacity  entropy"):
                        try:
                            f.readline()
                            f.readline()
                            f.readline()
                            f.readline()
                            f.readline()
                            line = f.readline().strip()
                            property_dict["Cv"] = float(line.split()[-3])
                        finally:
                            break
            if property_name == "energy":
                lines = f.readlines()
                for line in reversed(lines):
                    if "total E" in line:
                        property_dict["energy"] = float(line.split()[-1])
                        break
                    if "TOTAL ENERGY" in line:
                        property_dict["energy"] = float(line.split()[-3])
                        break

    property_score = property_dict[property_name]
    if property_score is None:
        raise ValueError(f"Cannot get {property_name}")
    return property_score


def run_xtb_command(positions, atom_numbers, acc=1.0, file_identifier="", property_name="force"):
    # write to .xyz file first
    mol_filename = f"mol_{file_identifier}.xyz"
    with open(mol_filename, 'w') as file:
        file.write(f"{len(atom_numbers)}\n\n")
        for atom_number, (x, y, z) in zip(atom_numbers, positions):
            atom_symbol = ATOM_NUMBER_TO_TYPE_DICT[atom_number.item()]
            file.write(f"{atom_symbol} {x:.5f} {y:.5f} {z:.5f}\n")
    if os.path.exists("gradient"):
        os.remove("gradient")
    if os.path.exists(f"mol_output_{file_identifier}.txt"):
        os.remove(f"mol_output_{file_identifier}.txt")
    # do xtb calculations
    command = f"xtb {mol_filename}"
    if property_name == "Cv":
        command += " --hess"
    command += f" --grad -a {str(acc)} -P 1"

    if file_identifier != "":
        command += f" --namespace {file_identifier}"
    command += f" > mol_output_{file_identifier}.txt"
    property_output = None
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        # print(f"{result.stdout.decode()}")
        if property_name == "force":
            property_output = gradient_from(file_identifier=file_identifier)
        else:
            property_output = property_from(property_name, file_identifier=file_identifier)
        property_output = torch.tensor(property_output)
    except subprocess.CalledProcessError as e:
        pass
    return property_output, file_identifier
