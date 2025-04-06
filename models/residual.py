import mlx.nn as nn


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )

    def __call__(self, x):
        return x + self.layers(x)


class ResidualStack:
    def __init__(self, in_dim, out_dim, h_dim, num_layers):
        self.stack = nn.Sequential(
            *[ResidualLayer(in_dim, out_dim, h_dim)] * num_layers
        )
        self.final = nn.ReLU()

    def __call__(self, x):
        x = self.stack(x)

        return self.final(x)
