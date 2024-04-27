import numpy as np
import torch.nn as nn
import torch
from prediction.vae_base import (
    Sampling,
    BaseVariationalAutoencoder,
    ExpandDims
)


def conv_output_size(L, kernel_size=3, stride=2, padding=1):
    return (L + 2 * padding - kernel_size) // stride + 1

class VariationalAutoencoderConv(BaseVariationalAutoencoder):

    stride = 2
    padding = 1
    kernel_size = 3
    output_padding = 1

    def __init__(
        self,
        encode_len,
        decode_len,
        feat_dim,
        latent_dim,
        hidden_layer_sizes,
        reconstruction_wt=5.0,
    ):
        super().__init__(
            encode_len, decode_len, feat_dim, latent_dim, reconstruction_wt
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _get_encoder(self):
        modules = []
        in_channels = self.feat_dim
        current_length = self.encode_len  # This is the initial length of the input
        for i, out_channels in enumerate(self.hidden_layer_sizes):
            modules.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            modules.append(nn.ReLU())
            in_channels = out_channels
            current_length = conv_output_size(current_length)

        modules.append(nn.Flatten())
        self.flat_features = in_channels * current_length
        self.fc1 = nn.Linear(self.flat_features, self.latent_dim)  # z_mean
        self.fc2 = nn.Linear(self.flat_features, self.latent_dim)  # z_log_var
        self.sampling = Sampling()
        return nn.Sequential(*modules)

    def _get_decoder(self):
        modules = []
        self.decoder_input = nn.Linear(
            self.latent_dim,
            self.flat_features,
        )
        modules.append(nn.Unflatten(1,(self.hidden_layer_sizes[-1],-1,),))
        num_layers = len(self.hidden_layer_sizes)
        for i in range(num_layers - 1, 0, -1):
            modules.append(
                nn.ConvTranspose1d(
                    self.hidden_layer_sizes[i],
                    self.hidden_layer_sizes[i-1],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                )
            )
            modules.append(nn.ReLU())
        # last de-convolution
        modules.append(
            nn.ConvTranspose1d(
                self.hidden_layer_sizes[0],
                1,  #univariate forecasting so exclude exog features
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
            )
        )
        modules.append(nn.ReLU())
        modules.append(nn.Flatten())
        final_conv_output_size = self.calculate_final_deconv_output_size()
        modules.append(nn.Linear(final_conv_output_size, self.decode_len))
        # expand dim to make it [N, decode_len, 1]
        modules.append(ExpandDims(dim=-1))

        return nn.Sequential(*modules)

    def forward(self, x):
        x = x.transpose(1, 2)  # [N, T, D] -> [N, D, T]
        x = self.encoder(x)
        z_mean = self.fc1(x)
        z_log_var = self.fc2(x)
        z = self.sampling(z_mean, z_log_var)
        x = self.decoder_input(z)
        reconstruction = self.decoder(x)
        return reconstruction, z_mean, z_log_var
    
    def calculate_final_deconv_output_size(self):
        current_size = self.flat_features // self.hidden_layer_sizes[-1]
        # Go through each transposed convolution layer
        num_layers = len(self.hidden_layer_sizes)
        for _ in range(num_layers - 1, -1, -1):
            # Apply the transposed convolution output size formula
            current_size = (self.stride * (current_size - 1) + \
                            self.kernel_size - 2 * self.padding + self.output_padding)
        return current_size


if __name__ == "__main__":
    N = 10
    T = 20
    D = 4
    encode_len = 12
    decode_len = T - encode_len

    model = VariationalAutoencoderConv(
        encode_len=encode_len,
        decode_len=decode_len,
        feat_dim=D,
        latent_dim=3,
        hidden_layer_sizes=[16, 32, 64],
    )
    # print(model)
    print(model.encoder)
    # print(model.decoder)
    # print(model.summary())

    X = np.random.rand(N, encode_len, D)
    reconstruction, z_mean, z_log_var = model(torch.tensor(X, dtype=torch.float32))
    print("out shape", reconstruction.shape)