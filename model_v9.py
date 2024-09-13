import torch
from math import log2
import torch.nn.functional as F
from spectral import SpectralNorm

from gigagan import *


class TextModifyEncoder(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.dim = dim

        self.learned_global_token = nn.Parameter(torch.randn(dim))

        self.project_in = nn.Identity()

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads
        )

    @beartype
    def forward(
        self,
        texts = None,
        text_encodings = None
    ):
        assert exists(texts) ^ exists(text_encodings)

        # mask = (text_encodings != 0.).any(dim = -1)
        mask = None
        text_encodings = self.project_in(text_encodings)

        # mask_with_global = F.pad(mask, (1, 0), value = True)
        mask_with_global = None

        batch = text_encodings.shape[0]
        global_tokens = repeat(self.learned_global_token, 'd -> b d', b = batch)

        text_encodings, ps = pack([global_tokens, text_encodings], 'b * d')

        text_encodings = self.transformer(text_encodings, mask = mask_with_global)

        global_tokens, text_encodings = unpack(text_encodings, ps, 'b * d')

        return global_tokens, text_encodings, mask


class gigan_generator(nn.Module):
    def __init__(
        self,
        image_size = 128,
        dim_capacity = 8,
        dim_max = 2048,
        channels = 3,
        style_network_dim = 512,
        style_network_depth = 4,
        dim_text_latent = 512,
        text_encoder_depth = 4,
        dim_latent = 512,
        self_attn_resolutions = (32, 16),
        self_attn_dim_head = 64,
        self_attn_heads = 8,
        self_attn_dot_product = True,
        self_attn_ff_mult = 4,
        cross_attn_resolutions = (32, 16),
        cross_attn_dim_head = 64,
        cross_attn_heads = 8,
        cross_attn_ff_mult = 4,
        num_conv_kernels = 2,  # the number of adaptive conv kernels
        num_skip_layers_excite = 4, #4
        unconditional = False,
        pixel_shuffle_upsample = False):
        super(gigan_generator, self).__init__()
        self.channels = channels
        self.style_network = StyleNetwork(dim = style_network_dim,depth = style_network_depth,
                                          dim_text_latent= dim_text_latent)

        self.text_encoder = TextModifyEncoder(dim = dim_text_latent, depth=text_encoder_depth)

        self.unconditional = unconditional

        assert not (unconditional and exists(text_encoder))
        assert not (unconditional and exists(style_network) and style_network.dim_text_latent > 0)

        assert is_power_of_two(image_size)
        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers

        is_adaptive = num_conv_kernels > 1
        dim_kernel_mod = num_conv_kernels if is_adaptive else 0

        style_embed_split_dims = []

        adaptive_conv = partial(AdaptiveConv2DMod, kernel = 3, num_conv_kernels = num_conv_kernels)

        # initial 4x4 block and conv

        self.init_block = nn.Parameter(torch.randn(dim_latent, 4, 4))
        self.init_conv = adaptive_conv(dim_latent, dim_latent)

        style_embed_split_dims.extend([
            dim_latent,
            dim_kernel_mod
        ])

        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers

        resolutions = image_size / ((2 ** torch.arange(num_layers).flip(0)))
        resolutions = resolutions.long().tolist()

        dim_layers = (2 ** (torch.arange(num_layers) + 1)) * dim_capacity
        dim_layers.clamp_(max = dim_max)

        dim_layers = torch.flip(dim_layers, (0,))
        dim_layers = F.pad(dim_layers, (1, 0), value = dim_latent)

        dim_layers = dim_layers.tolist()

        dim_pairs = list(zip(dim_layers[:-1], dim_layers[1:]))

        self.num_skip_layers_excite = num_skip_layers_excite

        self.layers = nn.ModuleList([])

        for ind, ((dim_in, dim_out), resolution) in enumerate(zip(dim_pairs, resolutions)):
            is_last = (ind + 1) == len(dim_pairs)
            is_first = ind == 0

            should_upsample = not is_first
            should_upsample_rgb = not is_last
            should_skip_layer_excite = num_skip_layers_excite > 0 and (ind + num_skip_layers_excite) < len(dim_pairs)

            has_self_attn = resolution in self_attn_resolutions
            has_cross_attn = resolution in cross_attn_resolutions and not unconditional

            skip_squeeze_excite = None
            if should_skip_layer_excite:
                dim_skip_in, _ = dim_pairs[ind + num_skip_layers_excite]
                skip_squeeze_excite = SqueezeExcite(dim_in, dim_skip_in)

            resnet_block = nn.ModuleList([
                adaptive_conv(dim_in, dim_out),
                Noise(dim_out),
                leaky_relu(),
                adaptive_conv(dim_out, dim_out),
                Noise(dim_out),
                leaky_relu()
            ])

            to_rgb = AdaptiveConv2DMod(dim_out, channels, 1, num_conv_kernels = 1, demod = False)

            self_attn = cross_attn = rgb_upsample = upsample = None
            
            upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

            upsample = upsample_klass(dim_in) if should_upsample else None
            rgb_upsample = upsample_klass(channels) if should_upsample_rgb else None

            if has_self_attn:
                self_attn = SelfAttentionBlock(
                    dim_out,
                    dim_head = self_attn_dim_head,
                    heads = self_attn_heads,
                    ff_mult = self_attn_ff_mult,
                    dot_product = self_attn_dot_product
            )

            if has_cross_attn:
                cross_attn = CrossAttentionBlock(
                    dim_out,
                    dim_context = dim_text_latent,
                    dim_head = cross_attn_dim_head,
                    heads = cross_attn_heads,
                    ff_mult = cross_attn_ff_mult,
                )

            style_embed_split_dims.extend([
                dim_in,             # for first conv in resnet block
                dim_kernel_mod,     # first conv kernel selection
                dim_out,            # second conv in resnet block
                dim_kernel_mod,     # second conv kernel selection
                dim_out,            # to RGB conv
                0,                  # RGB conv kernel selection
            ])

            self.layers.append(nn.ModuleList([
                skip_squeeze_excite,
                resnet_block,
                to_rgb,
                self_attn,
                cross_attn,
                upsample,
                rgb_upsample
            ]))

        # determine the projection of the style embedding to convolutional modulation weights (+ adaptive kernel selection weights) for all layers

        self.style_to_conv_modulations = nn.Linear(style_network_dim, sum(style_embed_split_dims))
        self.style_embed_split_dims = style_embed_split_dims

        self.apply(self.init_)
        nn.init.normal_(self.init_block, std = 0.02)

    def init_(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        styles = None,
        noise = None,
        # texts: List[str] | List[Tensor] | None = None,
        # text_encodings: Tensor | None = None,
        texts = None,
        text_encodings = None,
        global_text_tokens = None,
        fine_text_tokens = None,
        text_mask = None,
        batch_size = 1,
        return_all_rgbs = False
    ):

        if not self.unconditional:
            if exists(texts) or exists(text_encodings):
                assert exists(texts) ^ exists(text_encodings), 'either raw texts as List[str] or text_encodings (from clip) as Tensor is passed in, but not both'
                assert exists(self.text_encoder)

                if exists(texts):
                    global_text_tokens, fine_text_tokens, text_mask = self.text_encoder(texts = texts)
                elif exists(text_encodings):
                    global_text_tokens, fine_text_tokens, text_mask = self.text_encoder(text_encodings = text_encodings)

            else:
                assert all([*map(exists, (global_text_tokens, fine_text_tokens, text_mask))]), 'raw text or text embeddings were not passed in for conditional training'
        else:
            assert not any([*map(exists, (texts, global_text_tokens, fine_text_tokens))])

        styles = self.style_network(x = noise, text_latent= global_text_tokens)
        conv_mods = self.style_to_conv_modulations(styles)
        conv_mods = conv_mods.split(self.style_embed_split_dims, dim = -1)
        conv_mods = iter(conv_mods)

        batch_size = styles.shape[0]

        x = repeat(self.init_block, 'c h w -> b c h w', b = batch_size)
        x = self.init_conv(x, mod = next(conv_mods), kernel_mod = next(conv_mods))

        rgb = torch.zeros((batch_size, self.channels, 4, 4), device = self.device, dtype = x.dtype)

        # skip layer squeeze excitations

        excitations = [None] * self.num_skip_layers_excite

        # all the rgb's of each layer of the generator is to be saved for multi-resolution input discrimination

        rgbs = []

        # main network

        for squeeze_excite, (resnet_conv1, noise1, act1, resnet_conv2, noise2, act2), to_rgb_conv, self_attn, cross_attn, upsample, upsample_rgb in self.layers:

            if exists(upsample):
                x = upsample(x)

            if exists(squeeze_excite):
                skip_excite = squeeze_excite(x)
                excitations.append(skip_excite)

            excite = safe_unshift(excitations)
            if exists(excite):
                x = x * excite

            x = resnet_conv1(x, mod = next(conv_mods), kernel_mod = next(conv_mods))
            x = noise1(x)
            x = act1(x)

            x = resnet_conv2(x, mod = next(conv_mods), kernel_mod = next(conv_mods))
            x = noise2(x)
            x = act2(x)

            if exists(self_attn):
                x = self_attn(x)

            if exists(cross_attn):
                x = cross_attn(x, context = fine_text_tokens, mask = text_mask)
                # x = cross_attn(x, context = global_text_tokens, mask = text_mask)

            layer_rgb = to_rgb_conv(x, mod = next(conv_mods), kernel_mod = next(conv_mods))

            rgb = rgb + layer_rgb

            rgbs.append(rgb)

            if exists(upsample_rgb):
                rgb = upsample_rgb(rgb)

        # sanity check

        assert is_empty([*conv_mods]), 'convolutions were incorrectly modulated'

        if return_all_rgbs:
            return rgb, rgbs

        return rgb

class sa_encoder(nn.Module):
    def __init__(self, in_channels, dec_channels, latent_size,
                self_attn_dim_head = 64,
                self_attn_heads = 8,
                self_attn_dot_product = True,
                self_attn_ff_mult = 4,):
        super(sa_encoder, self).__init__()
        self.dec_channels = dec_channels
        self.e_attn1 = SelfAttentionBlock(
                    dec_channels*8,
                    dim_head = self_attn_dim_head,
                    heads = self_attn_heads,
                    ff_mult = self_attn_ff_mult,
                    dot_product = self_attn_dot_product
            )

        self.e_attn2 = SelfAttentionBlock(
                    dec_channels*16,
                    dim_head = self_attn_dim_head,
                    heads = self_attn_heads,
                    ff_mult = self_attn_ff_mult,
                    dot_product = self_attn_dot_product
            )
        self.e_conv1 = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels, dec_channels,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        # 64x64
        self.e_conv2 = nn.Sequential(SpectralNorm(nn.Conv2d(dec_channels, dec_channels*2,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        #32x32
        self.e_conv3 = nn.Sequential(SpectralNorm(nn.Conv2d(dec_channels*2, dec_channels*4,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        #16x16
        self.e_conv4 = nn.Sequential(SpectralNorm(nn.Conv2d(dec_channels*4, dec_channels*8,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        #8x8
        self.e_conv5 = nn.Sequential(SpectralNorm(nn.Conv2d(dec_channels*8, dec_channels*16,
                                               kernel_size=(4,4), stride=2, padding=1)),
                                    nn.LeakyReLU())
        #4x4
        
        self.e_fc_1 = nn.Linear(dec_channels*16*4*4, latent_size)
        

    def forward(self, x):
        x = self.e_conv1(x)
        x = self.e_conv2(x)
        x = self.e_conv3(x)
        x = self.e_conv4(x)
        x = self.e_attn1(x)
        x = self.e_conv5(x)
        x = self.e_attn2(x)
        x = x.view((-1, self.dec_channels*16*4*4))
        z = self.e_fc_1(x)
        return z

    
class GigaGAN(nn.Module):
    def __init__(self, in_channels, dec_channels, latent_size, private_classes):
        super(GigaGAN, self).__init__()
        self.decoder = gigan_generator()
        self.encoder = sa_encoder(in_channels, dec_channels, latent_size)

        self.private_embedding = nn.Linear(latent_size, private_classes)
        
        self.private_latent = self.private_embedding.weight
        self.tanh = nn.Tanh()
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z, y, is_private):
        bs = z.size(0)
        if is_private:
            z_priv = self.private_latent.mean(dim=0).squeeze(0).repeat((bs,1))
        else:
            z_priv = self.private_latent[[y]].view((bs,-1))
        z_priv = torch.unsqueeze(z_priv, 1)
        recon = self.decoder(noise = z, texts = None, text_encodings = z_priv)
        # recon = self.tanh(recon)
        # recon = self.tanh(recon)
        return recon
    
    def forward(self, x, y, is_private=False):
        latent = self.encode(x)
        recon = self.decode(latent, y, is_private)
        return latent, recon

class DisNet(nn.Module):
    def __init__(self, latent_size, hidden_channels, private_classes):
        super(DisNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(latent_size, hidden_channels*16),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(hidden_channels*16))
        
        self.fc2 = nn.Sequential(nn.Linear(hidden_channels*16, hidden_channels*8),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(hidden_channels*8))
        
        self.fc3 = nn.Linear(hidden_channels*8, private_classes)
                                 
        
    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        return self.fc3(y)
