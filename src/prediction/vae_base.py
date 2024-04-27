import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def forward(self, z_mean, z_log_var):
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class ExpandDims(nn.Module):
    def __init__(self, dim):
        super(ExpandDims, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

class BaseVariationalAutoencoder(nn.Module, ABC):
    def __init__(
        self, encode_len, decode_len, feat_dim, latent_dim, reconstruction_wt=5.0
    ):
        super().__init__()
        self.encode_len = encode_len
        self.decode_len = decode_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        # Define metrics
        self.total_loss_tracker = torch.tensor(0.0)
        self.reconstruction_loss_tracker = torch.tensor(0.0)
        self.kl_loss_tracker = torch.tensor(0.0)
        # Initialize encoders and decoders to None
        self.encoder = None
        self.decoder = None

    def forward(self, X):
        # This needs to be implemented based on the encoder and decoder structure
        pass

    @abstractmethod
    def _get_encoder(self):
        pass

    @abstractmethod
    def _get_decoder(self):
        pass

    def summary(self):
        print(self.encoder)
        print(self.decoder)
