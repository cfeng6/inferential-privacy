from __future__ import annotations
from omegaconf import DictConfig

from collections import namedtuple
from pathlib import Path
from math import log2, sqrt
from random import random
from functools import partial

from torchvision import utils

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.autograd import grad as torch_grad
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
 
from open_clip import OpenClipAdapter

# from gigagan_pytorch import __BEARTYPE__
# if __BEARTYPE__:
    # from beartype import beartype
    # from beartype.typing import List, Tuple, Dict, Iterable
# else:
def beartype(fn, *args, **karg):
    return fn
from typing import List, Tuple, Dict, Iterable

from einops import rearrange, pack, unpack, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from kornia.filters import filter2d

from ema_pytorch import EMA
from tqdm import tqdm

from numerize import numerize

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs


def exists(val):
    return val is not None

@beartype
def is_empty(arr: Iterable):
    return len(arr) == 0

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def is_power_of_two(n):
    return log2(n).is_integer()

def safe_unshift(arr):
    if len(arr) == 0:
        return None
    return arr.pop(0)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def group_by_num_consecutive(arr, num):
    out = []
    for ind, el in enumerate(arr):
        if ind > 0 and divisible_by(ind, num):
            yield out
            out = []

        out.append(el)

    if len(out) > 0:
        yield out

def is_unique(arr):
    return len(set(arr)) == len(arr)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups, remainder = divmod(num, divisor)
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def mkdir_if_not_exists(path):
    path.mkdir(exist_ok = True, parents = True)

@beartype
def set_requires_grad_(
    m: nn.Module,
    requires_grad: bool
):
    for p in m.parameters():
        p.requires_grad = requires_grad

# activation functions

def leaky_relu(neg_slope = 0.2):
    return nn.LeakyReLU(neg_slope)

def conv2d_3x3(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 3, padding = 1)

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gradient_penalty(
    images,
    outputs,
    grad_output_weights = None,
    weight = 10,
    scaler: GradScaler | None = None,
    eps = 1e-4
):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    if exists(scaler):
        outputs = [*map(scaler.scale, outputs)]

    if not exists(grad_output_weights):
        grad_output_weights = (1,) * len(outputs)

    maybe_scaled_gradients, *_ = torch_grad(
        outputs = outputs,
        inputs = images,
        grad_outputs = [(torch.ones_like(output) * weight) for output, weight in zip(outputs, grad_output_weights)],
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )

    gradients = maybe_scaled_gradients

    if exists(scaler):
        scale = scaler.get_scale()
        inv_scale = 1. / max(scale, eps)
        gradients = maybe_scaled_gradients * inv_scale

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

# hinge gan losses

def generator_hinge_loss(fake):
    return fake.mean()

def discriminator_hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

# auxiliary losses

def aux_matching_loss(real, fake):
    """
    making logits negative, as in this framework, discriminator is 0 for real, high value for fake. GANs can have this arbitrarily swapped, as it only matters if the generator and discriminator are opposites
    """
    return (log(1 + (-real).exp()) + log(1 + (-fake).exp())).mean()

@beartype
def aux_clip_loss(
    clip: OpenClipAdapter,
    images: Tensor,
    texts: List[str] | None = None,
    text_embeds: Tensor | None = None
):
    assert exists(texts) ^ exists(text_embeds)

    images, batch_sizes = all_gather(images, 0, None)

    if exists(texts):
        text_embeds, _ = clip.embed_texts(texts)
        text_embeds, _ = all_gather(text_embeds, 0, batch_sizes)

    return clip.contrastive_loss(images = images, text_embeds = text_embeds)

# differentiable augmentation - Karras et al. stylegan-ada
# start with horizontal flip

class DiffAugment(nn.Module):
    def __init__(
        self,
        *,
        prob,
        horizontal_flip,
        horizontal_flip_prob = 0.5
    ):
        super().__init__()
        self.prob = prob
        assert 0 <= prob <= 1.

        self.horizontal_flip = horizontal_flip
        self.horizontal_flip_prob = horizontal_flip_prob

    def forward(
        self,
        images,
        rgbs: List[Tensor]
    ):
        if random() >= self.prob:
            return images, rgbs

        if random() < self.horizontal_flip_prob:
            images = torch.flip(images, (-1,))
            rgbs = [torch.flip(rgb, (-1,)) for rgb in rgbs]

        return images, rgbs

# rmsnorm (newer papers show mean-centering in layernorm not necessary)

class ChannelRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        normed = F.normalize(x, dim = 1)
        return normed * self.scale * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# down and upsample

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized = True)

def Upsample(*args):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
        Blur()
    )

class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim):
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim, 1)
    )

# skip layer excitation

def SqueezeExcite(dim, dim_out, reduction = 4, dim_min = 32):
    dim_hidden = max(dim_out // reduction, dim_min)

    return nn.Sequential(
        Reduce('b c h w -> b c', 'mean'),
        nn.Linear(dim, dim_hidden),
        nn.SiLU(),
        nn.Linear(dim_hidden, dim_out),
        nn.Sigmoid(),
        Rearrange('b c -> b c 1 1')
    )

# adaptive conv
# the main novelty of the paper - they propose to learn a softmax weighted sum of N convolutional kernels, depending on the text embedding

def get_same_padding(size, kernel, dilation, stride):
    return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

class AdaptiveConv2DMod(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        demod = True,
        stride = 1,
        dilation = 1,
        eps = 1e-8,
        num_conv_kernels = 1 # set this to be greater than 1 for adaptive
    ):
        super().__init__()
        self.eps = eps

        self.dim_out = dim_out

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1

        self.weights = nn.Parameter(torch.randn((num_conv_kernels, dim_out, dim, kernel, kernel)))

        self.demod = demod

        nn.init.kaiming_normal_(self.weights, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def forward(
        self,
        fmap,
        mod: Tensor,
        kernel_mod: Tensor | None = None
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b, h = fmap.shape[0], fmap.shape[-2]

        # account for feature map that has been expanded by the scale in the first dimension
        # due to multiscale inputs and outputs

        if mod.shape[0] != b:
            mod = repeat(mod, 'b ... -> (s b) ...', s = b // mod.shape[0])

        if exists(kernel_mod):
            kernel_mod_has_el = kernel_mod.numel() > 0

            assert self.adaptive or not kernel_mod_has_el

            if kernel_mod_has_el and kernel_mod.shape[0] != b:
                kernel_mod = repeat(kernel_mod, 'b ... -> (s b) ...', s = b // kernel_mod.shape[0])

        # prepare weights for modulation

        weights = self.weights

        if self.adaptive:
            weights = repeat(weights, '... -> b ...', b = b)

            # determine an adaptive weight and 'select' the kernel to use with softmax

            assert exists(kernel_mod) and kernel_mod.numel() > 0

            kernel_attn = kernel_mod.softmax(dim = -1)
            kernel_attn = rearrange(kernel_attn, 'b n -> b n 1 1 1 1')

            weights = reduce(weights * kernel_attn, 'b n ... -> b ...', 'sum')

        # do the modulation, demodulation, as done in stylegan2

        mod = rearrange(mod, 'b i -> b 1 i 1 1')

        weights = weights * (mod + 1)

        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k1 k2 -> b o 1 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c h w -> 1 (b c) h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        padding = get_same_padding(h, self.kernel, self.dilation, self.stride)
        fmap = F.conv2d(fmap, weights, padding = padding, groups = b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)
    
class AdaptiveConv1DMod(nn.Module):
    """ 1d version of adaptive conv, for time dimension in videogigagan """

    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        demod = True,
        stride = 1,
        dilation = 1,
        eps = 1e-8,
        num_conv_kernels = 1 # set this to be greater than 1 for adaptive
    ):
        super().__init__()
        self.eps = eps

        self.dim_out = dim_out

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1

        self.weights = nn.Parameter(torch.randn((num_conv_kernels, dim_out, dim, kernel)))

        self.demod = demod

        nn.init.kaiming_normal_(self.weights, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def forward(
        self,
        fmap,
        mod: Tensor,
        kernel_mod: Tensor | None = None
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b, t = fmap.shape[0], fmap.shape[-1]

        # account for feature map that has been expanded by the scale in the first dimension
        # due to multiscale inputs and outputs

        if mod.shape[0] != b:
            mod = repeat(mod, 'b ... -> (s b) ...', s = b // mod.shape[0])

        if exists(kernel_mod):
            kernel_mod_has_el = kernel_mod.numel() > 0

            assert self.adaptive or not kernel_mod_has_el

            if kernel_mod_has_el and kernel_mod.shape[0] != b:
                kernel_mod = repeat(kernel_mod, 'b ... -> (s b) ...', s = b // kernel_mod.shape[0])

        # prepare weights for modulation

        weights = self.weights

        if self.adaptive:
            weights = repeat(weights, '... -> b ...', b = b)

            # determine an adaptive weight and 'select' the kernel to use with softmax

            assert exists(kernel_mod) and kernel_mod.numel() > 0

            kernel_attn = kernel_mod.softmax(dim = -1)
            kernel_attn = rearrange(kernel_attn, 'b n -> b n 1 1 1')

            weights = reduce(weights * kernel_attn, 'b n ... -> b ...', 'sum')

        # do the modulation, demodulation, as done in stylegan2

        mod = rearrange(mod, 'b i -> b 1 i 1')

        weights = weights * (mod + 1)

        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k -> b o 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c t -> 1 (b c) t')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        padding = get_same_padding(t, self.kernel, self.dilation, self.stride)
        fmap = F.conv1d(fmap, weights, padding = padding, groups = b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)
    
# attention
# they use an attention with a better Lipchitz constant - l2 distance similarity instead of dot product - also shared query / key space - shown in vitgan to be more stable
# not sure what they did about token attention to self, so masking out, as done in some other papers using shared query / key space

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dot_product = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.dot_product = dot_product

        self.norm = ChannelRMSNorm(dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_inner, 1, bias = False) if dot_product else None
        self.to_v = nn.Conv2d(dim, dim_inner, 1, bias = False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias = False)

    def forward(self, fmap):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = fmap.shape[0]

        fmap = self.norm(fmap)

        x, y = fmap.shape[-2:]

        h = self.heads

        q, v = self.to_q(fmap), self.to_v(fmap)

        k = self.to_k(fmap) if exists(self.to_k) else q

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = self.heads), (q, k, v))

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b = batch), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # l2 distance or dot product

        if self.dot_product:
            sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            # using pytorch cdist leads to nans in lightweight gan training framework, at least
            q_squared = (q * q).sum(dim = -1)
            k_squared = (k * k).sum(dim = -1)
            l2dist_squared = rearrange(q_squared, 'b i -> b i 1') + rearrange(k_squared, 'b j -> b 1 j') - 2 * einsum('b i d, b j d -> b i j', q, k) # hope i'm mathing right
            sim = -l2dist_squared

        # scale

        sim = sim * self.scale

        # attention

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)

        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        kv_input_dim = default(dim_context, dim)

        self.norm = ChannelRMSNorm(dim)
        self.norm_context = RMSNorm(kv_input_dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias = False)
        self.to_kv = nn.Linear(kv_input_dim, dim_inner * 2, bias = False)
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias = False)

    def forward(self, fmap, context, mask = None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """

        fmap = self.norm(fmap)
        context = self.norm_context(context)

        x, y = fmap.shape[-2:]

        h = self.heads

        q, k, v = (self.to_q(fmap), *self.to_kv(context).chunk(2, dim = -1))
        
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (k, v))

        q = rearrange(q, 'b (h d) x y -> (b h) (x y) d', h = self.heads)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = repeat(mask, 'b j -> (b h) 1 j', h = self.heads)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)

        return self.to_out(out)

# classic transformer attention, stick with l2 distance

class TextAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, encodings, mask = None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = encodings.shape[0]

        encodings = self.norm(encodings)

        h = self.heads

        q, k, v = self.to_qkv(encodings).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b = batch), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # key padding mask

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = repeat(mask, 'b n -> (b h) 1 n', h = h)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        return self.to_out(out)

# feedforward

def FeedForward(
    dim,
    mult = 4,
    channel_first = False
):
    dim_hidden = int(dim * mult)
    norm_klass = ChannelRMSNorm if channel_first else RMSNorm
    proj = partial(nn.Conv2d, kernel_size = 1) if channel_first else nn.Linear

    return nn.Sequential(
        norm_klass(dim),
        proj(dim, dim_hidden),
        nn.GELU(),
        proj(dim_hidden, dim)
    )

# different types of transformer blocks or transformers (multiple blocks)

class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        dot_product = False
    ):
        super().__init__()
        self.attn = SelfAttention(dim = dim, dim_head = dim_head, heads = heads, dot_product = dot_product)
        self.ff = FeedForward(dim = dim, mult = ff_mult, channel_first = True)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.attn = CrossAttention(dim = dim, dim_context = dim_context, dim_head = dim_head, heads = heads)
        self.ff = FeedForward(dim = dim, mult = ff_mult, channel_first = True)

    def forward(self, x, context, mask = None):
        x = self.attn(x, context = context, mask = mask) + x
        x = self.ff(x) + x
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TextAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.norm(x)

# text encoder

class TextEncoder(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        depth,
        clip: OpenClipAdapter | None = None,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.dim = dim

        if not exists(clip):
            clip = OpenClipAdapter()

        self.clip = clip
        set_requires_grad_(clip, False)

        self.learned_global_token = nn.Parameter(torch.randn(dim))

        self.project_in = nn.Linear(clip.dim_latent, dim) if clip.dim_latent != dim else nn.Identity()

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads
        )

    @beartype
    def forward(
        self,
        texts: List[str] | None = None,
        text_encodings: Tensor | None = None
    ):
        assert exists(texts) ^ exists(text_encodings)

        if not exists(text_encodings):
            with torch.no_grad():
                self.clip.eval()
                _, text_encodings = self.clip.embed_texts(texts)

        mask = (text_encodings != 0.).any(dim = -1)

        text_encodings = self.project_in(text_encodings)

        mask_with_global = F.pad(mask, (1, 0), value = True)

        batch = text_encodings.shape[0]
        global_tokens = repeat(self.learned_global_token, 'd -> b d', b = batch)

        text_encodings, ps = pack([global_tokens, text_encodings], 'b * d')

        text_encodings = self.transformer(text_encodings, mask = mask_with_global)

        global_tokens, text_encodings = unpack(text_encodings, ps, 'b * d')

        return global_tokens, text_encodings, mask

# style mapping network

class EqualLinear(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        lr_mul = 1,
        bias = True
    ):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleNetwork(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        lr_mul = 0.1,
        dim_text_latent = 0
    ):
        super(StyleNetwork, self).__init__()
        self.dim = dim
        self.dim_text_latent = dim_text_latent

        layers = []
        for i in range(depth):
            is_first = i == 0
            dim_in = (dim + dim_text_latent) if is_first else dim

            layers.extend([EqualLinear(dim_in, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x,
        text_latent = None
    ):
        x = F.normalize(x, dim = 1)

        if self.dim_text_latent > 0:
            assert exists(text_latent)
            x = torch.cat((x, text_latent), dim = -1)

        return self.net(x)

# noise

class Noise(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(
        self,
        x,
        noise = None
    ):
        b, _, h, w, device = *x.shape, x.device

        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device = device)

        return x + self.weight * noise

@beartype
class SimpleDecoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dims: Tuple[int, ...],
        patch_dim: int = 1,
        frac_patches: float = 1.,
        dropout: float = 0.5
    ):
        super().__init__()
        assert 0 < frac_patches <= 1.

        self.patch_dim = patch_dim
        self.frac_patches = frac_patches

        self.dropout = nn.Dropout(dropout)

        dims = [dim, *dims]

        layers = [conv2d_3x3(dim, dim)]

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Sequential(
                Upsample(dim_in),
                conv2d_3x3(dim_in, dim_out),
                leaky_relu()
            ))

        self.net = nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        fmap,
        orig_image
    ):
        fmap = self.dropout(fmap)

        if self.frac_patches < 1.:
            batch, patch_dim = fmap.shape[0], self.patch_dim
            fmap_size, img_size = fmap.shape[-1], orig_image.shape[-1]

            assert divisible_by(fmap_size, patch_dim), f'feature map dimensions are {fmap_size}, but the patch dim was designated to be {patch_dim}'
            assert divisible_by(img_size, patch_dim), f'image size is {img_size} but the patch dim was specified to be {patch_dim}'

            fmap, orig_image = map(lambda t: rearrange(t, 'b c (p1 h) (p2 w) -> b (p1 p2) c h w', p1 = patch_dim, p2 = patch_dim), (fmap, orig_image))

            total_patches = patch_dim ** 2
            num_patches_recon = max(int(self.frac_patches * total_patches), 1)

            batch_arange = torch.arange(batch, device = self.device)[..., None]
            batch_randperm = torch.randn((batch, total_patches)).sort(dim = -1).indices
            patch_indices = batch_randperm[..., :num_patches_recon]

            fmap, orig_image = map(lambda t: t[batch_arange, patch_indices], (fmap, orig_image))
            fmap, orig_image = map(lambda t: rearrange(t, 'b p ... -> (b p) ...'), (fmap, orig_image))

        recon = self.net(fmap)
        return F.mse_loss(recon, orig_image)

class RandomFixedProjection(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        channel_first = True
    ):
        super().__init__()
        weights = torch.randn(dim, dim_out)
        nn.init.kaiming_normal_(weights, mode = 'fan_out', nonlinearity = 'linear')

        self.channel_first = channel_first
        self.register_buffer('fixed_weights', weights)

    def forward(self, x):
        if not self.channel_first:
            return x @ self.fixed_weights

        return einsum('b c ..., c d -> b d ...', x, self.fixed_weights)

class VisionAidedDiscriminator(nn.Module):
    """ the vision-aided gan loss """

    @beartype
    def __init__(
        self,
        *,
        depth = 2,
        dim_head = 64,
        heads = 8,
        clip: OpenClipAdapter | None = None,
        layer_indices = (-1, -2, -3),
        conv_dim = None,
        text_dim = None,
        unconditional = False,
        num_conv_kernels = 2
    ):
        super().__init__()

        if not exists(clip):
            clip = OpenClipAdapter()

        self.clip = clip
        dim = clip._dim_image_latent

        self.unconditional = unconditional
        text_dim = default(text_dim, dim)
        conv_dim = default(conv_dim, dim)

        self.layer_discriminators = nn.ModuleList([])
        self.layer_indices = layer_indices

        conv_klass = partial(AdaptiveConv2DMod, kernel = 3, num_conv_kernels = num_conv_kernels) if not unconditional else conv2d_3x3

        for _ in layer_indices:
            self.layer_discriminators.append(nn.ModuleList([
                RandomFixedProjection(dim, conv_dim),
                conv_klass(conv_dim, conv_dim),
                nn.Linear(text_dim, conv_dim) if not unconditional else None,
                nn.Linear(text_dim, num_conv_kernels) if not unconditional else None,
                nn.Sequential(
                    conv2d_3x3(conv_dim, 1),
                    Rearrange('b 1 ... -> b ...')
                )
            ]))

    def parameters(self):
        return self.layer_discriminators.parameters()

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    @beartype
    def forward(
        self,
        images,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        return_clip_encodings = False
    ):

        assert self.unconditional or (exists(text_embeds) ^ exists(texts))

        with torch.no_grad():
            if not self.unconditional and exists(texts):
                self.clip.eval()
                text_embeds = self.clip.embed_texts

        _, image_encodings = self.clip.embed_images(images)

        logits = []

        for layer_index, (rand_proj, conv, to_conv_mod, to_conv_kernel_mod, to_logits) in zip(self.layer_indices, self.layer_discriminators):
            image_encoding = image_encodings[layer_index]

            cls_token, rest_tokens = image_encoding[:, :1], image_encoding[:, 1:]
            height_width = int(sqrt(rest_tokens.shape[-2])) # assume square

            img_fmap = rearrange(rest_tokens, 'b (h w) d -> b d h w', h = height_width)

            img_fmap = img_fmap + rearrange(cls_token, 'b 1 d -> b d 1 1 ') # pool the cls token into the rest of the tokens

            img_fmap = rand_proj(img_fmap)

            if self.unconditional:
                img_fmap = conv(img_fmap)
            else:
                assert exists(text_embeds)

                img_fmap = conv(
                    img_fmap,
                    mod = to_conv_mod(text_embeds),
                    kernel_mod = to_conv_kernel_mod(text_embeds)
                )

            layer_logits = to_logits(img_fmap)

            logits.append(layer_logits)

        if not return_clip_encodings:
            return logits

        return logits, image_encodings

class Predictor(nn.Module):
    def __init__(
        self,
        dim,
        depth = 4,
        num_conv_kernels = 2,
        unconditional = False
    ):
        super().__init__()
        self.unconditional = unconditional
        self.residual_fn = nn.Conv2d(dim, dim, 1)
        self.residual_scale = 2 ** -0.5

        self.layers = nn.ModuleList([])

        klass = nn.Conv2d if unconditional else partial(AdaptiveConv2DMod, num_conv_kernels = num_conv_kernels)
        klass_kwargs = dict(padding = 1) if unconditional else dict()

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                klass(dim, dim, 3, **klass_kwargs),
                leaky_relu(),
                klass(dim, dim, 3, **klass_kwargs),
                leaky_relu()
            ]))

        self.to_logits = nn.Conv2d(dim, 1, 1)

    def forward(
        self,
        x,
        mod = None,
        kernel_mod = None
    ):
        residual = self.residual_fn(x)

        kwargs = dict()

        if not self.unconditional:
            kwargs = dict(mod = mod, kernel_mod = kernel_mod)

        for conv1, activation, conv2, activation in self.layers:

            inner_residual = x

            x = conv1(x, **kwargs)
            x = activation(x)
            x = conv2(x, **kwargs)
            x = activation(x)

            x = x + inner_residual
            x = x * self.residual_scale

        x = x + residual
        return self.to_logits(x)

