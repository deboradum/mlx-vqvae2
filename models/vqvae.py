import mlx.nn as nn
import mlx.core as mx

from models.encoder import Top_encoder, Bottom_encoder
from models.decoder import Top_decoder, Decoder
from models.quantizer import Quantizer


class VQVAE2(nn.Module):
    def __init__(self, encoder_h_dim, res_h_dim, num_res_layers, k, d, beta):
        super().__init__()

        self.encoder_top = Top_encoder(
            encoder_h_dim, encoder_h_dim, res_h_dim, num_res_layers
        )
        self.pre_quantization_conv_top = nn.Conv2d(encoder_h_dim, d, kernel_size=1)
        self.quantizer_top = Quantizer(k, d, beta)
        self.decoder_top = Top_decoder(d, encoder_h_dim, d, res_h_dim, num_res_layers)
        self.top_upscale = nn.ConvTranspose2d(d, d, 4, stride=2, padding=1)

        self.encoder_btm = Bottom_encoder(
            3, encoder_h_dim, res_h_dim, num_res_layers
        )
        self.pre_quantization_conv_btm = nn.Conv2d(encoder_h_dim+d, d, kernel_size=1)
        self.quantizer_btm = Quantizer(k, d, beta)

        self.decoder = Decoder(2*d, encoder_h_dim, 3, res_h_dim, num_res_layers)

    def __call__(self, x):
        # the encoder network first transforms and downsamples the image by a factor of 4 ...
        h_btm = self.encoder_btm(x)
        # ... Another stack of residual blocks then further scales down the representations by a factor of two.
        h_top = self.encoder_top(h_btm)

        # Top part of fig 2a.
        h_top = self.pre_quantization_conv_top(h_top)
        top_loss, e_top, perplexity = self.quantizer_top(h_top)
        dec_top = self.decoder_top(h_top)

        # Middle part of fig 2a.
        h_btm = self.pre_quantization_conv_btm(mx.concatenate((h_btm, dec_top), axis=-1))
        btm_loss, e_btm, perplexity = self.quantizer_btm(h_btm)

        # Upsample e_top so it can be concatenated with e_btm before final decoder.
        e_top = self.top_upscale(e_top)

        # The decoder ...[] takes as input all levels of the quantized latent hierarchy
        x_hat = self.decoder(mx.concatenate((e_btm, e_top), axis=-1))

        return x_hat, top_loss, btm_loss, perplexity
