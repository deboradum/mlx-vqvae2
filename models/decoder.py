import mlx.nn as nn

from models.residual import ResidualStack


class Top_decoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, res_h_dim, num_res_layers):
        super().__init__()

        self.inv_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, num_res_layers),
            nn.ConvTranspose2d(h_dim, out_dim, kernel_size=4, stride=2, padding=1),
        )

    def __call__(self, x):
        return self.inv_conv_layers(x)

class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, res_h_dim, num_res_layers):
        super().__init__()

        self.inv_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, num_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, out_dim, kernel_size=4, stride=2, padding=1),
        )

    def __call__(self, x):
        return self.inv_conv_layers(x)
