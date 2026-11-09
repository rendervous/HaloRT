"""
halort._modeling - minimal model definitions for Denoiser, DenoiserX and
HaloRT (ScatNet-family torch.nn.Module classes), extracted from the
direct_neural_rendering project's own rdv_nr/_modeling.py. Diffusion-model
classes, unused profiler variants, and other rdv_nr content not needed by
these three architectures were dropped - see create_scatnet() in
halort/__init__.py for the intended entry point.

Needs the `rdv` package (Vulkan-CUDA interop rendering engine) installed
separately - see the top-level README for setup instructions.
"""
import torch
import numpy as np
import typing
from typing import Any
import math
from torch.nn import Module, ModuleList
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange
from functools import partial, wraps
from packaging import version
from collections import namedtuple
import rdv


def unwrap_model_parameters(d):
    new_state = {}
    for k, v in d.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )


def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        # Plain 1D (not (1, dim, 1, 1)) - DDP's gradient reducer (under
        # torch.compile) rejects a (1, dim, 1, 1)-shaped parameter's
        # gradient when the compiled backward hands it back canonicalized
        # to (dim,): "returned an invalid gradient ... expected shape
        # compatible with [1, C, 1, 1]" - confirmed neither
        # torch._dynamo.config.optimize_ddp=False nor DDP's
        # gradient_as_bucket_view=False avoid this; storing the parameter
        # 1D to begin with (reshaping only for the broadcast below) sidesteps
        # it at the source instead of working around it.
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g.view(1, -1, 1, 1) * self.scale


class Block(Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def _sdp_backends(config):
    """AttentionConfig's own (enable_flash, enable_math, enable_mem_efficient)
    booleans, translated to the backend list torch.nn.attention.sdpa_kernel
    expects (replaces the deprecated torch.backends.cuda.sdp_kernel)."""
    backends = []
    if config.enable_flash:
        backends.append(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
    if config.enable_mem_efficient:
        backends.append(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
    if config.enable_math:
        backends.append(torch.nn.attention.SDPBackend.MATH)
    return backends


def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner


print_once = once(print)


class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        # Originally hardcoded to major==8/minor==0 (A100 only) - overly
        # conservative from flash-attention's early rollout, when kernel
        # support across GPUs was spotty. Flash attention has been
        # confirmed to run correctly (same results, faster) on every
        # Ampere-or-later card actually available on this cluster,
        # including compute capability 12.0 (RTX 5090) - restricting to
        # exactly 8.0 silently forced ~2x slower math/mem-efficient
        # attention on all of them. Enabling all three (not flash-only):
        # flash has no fp32 kernel at all ("No available kernel" - see
        # benchmarking_nn_fp16.py's own fp32 eager path), so math/mem-
        # efficient must stay available as a fallback for non-fp16/bf16
        # callers; sdpa_kernel still prefers flash whenever the actual
        # call's dtype/shape supports it.
        if device_properties.major >= 8:
            print_once(f'GPU (compute {device_properties.major}.{device_properties.minor}) supports flash attention, enabling it alongside the other backends')
            self.cuda_config = AttentionConfig(True, True, True)
        else:
            print_once('Pre-Ampere GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.nn.attention.sdpa_kernel(_sdp_backends(config)):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class BlockAttention(Module):
    """
    Local block attention with overlap padding.

    Input:
        x : [B, C, H, W]

    Procedure:
        1. Split image into blocks of size (block_h + 2*pad_h, block_w + 2*pad_w)
        2. Apply attention independently inside each block
        3. Keep only the central (block_h, block_w) region
        4. Stitch blocks back to the original HxW grid

    This behaves similarly to:
        - window attention
        - halo attention
        - local transformer attention
    """

    def __init__(
        self,
        dim,
        block_size=(8, 8),
        pad_size=(2, 2),
        heads=4,
        dim_head=32,
        num_mem_kv=4,
        flash=False
    ):
        super().__init__()

        self.heads = heads
        self.block_h, self.block_w = block_size
        self.pad_h, self.pad_w = pad_size

        hidden_dim = heads * dim_head

        self.norm = RMSNorm(dim)

        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(
            torch.randn(2, heads, num_mem_kv, dim_head)
        )

        self.to_qkv = nn.Conv2d(
            dim,
            hidden_dim * 3,
            kernel_size=1,
            bias=False
        )

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):

        b, c, H, W = x.shape

        bh, bw = self.block_h, self.block_w
        ph, pw = self.pad_h, self.pad_w

        assert H % bh == 0
        assert W % bw == 0

        x = self.norm(x)

        # ----------------------------------------------------------
        # pad globally first
        # ----------------------------------------------------------

        x = F.pad(
            x,
            (pw, pw, ph, ph),
            mode='reflect'
        )

        # padded spatial size
        Hp = H + 2 * ph
        Wp = W + 2 * pw

        # ----------------------------------------------------------
        # qkv
        # ----------------------------------------------------------

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # ----------------------------------------------------------
        # extract overlapping blocks
        # each block sees halo/context
        # ----------------------------------------------------------

        kernel_h = bh + 2 * ph
        kernel_w = bw + 2 * pw

        stride = (bh, bw)

        def unfold_block(t):

            # [B, C*K*K, N]
            t = F.unfold(
                t,
                kernel_size=(kernel_h, kernel_w),
                stride=stride
            )

            # number of blocks
            n_blocks = t.shape[-1]

            # [B, C, Kh, Kw, N]
            t = rearrange(
                t,
                'b (c kh kw) n -> b n c kh kw',
                kh=kernel_h,
                kw=kernel_w
            )

            return t

        q = unfold_block(q)
        k = unfold_block(k)
        v = unfold_block(v)

        # number of blocks
        n_blocks = q.shape[1]

        # ----------------------------------------------------------
        # flatten local windows
        # ----------------------------------------------------------

        def to_attn(t):

            return rearrange(
                t,
                'b n (h d) kh kw -> (b n) h (kh kw) d',
                h=self.heads
            )

        q = to_attn(q)
        k = to_attn(k)
        v = to_attn(v)

        # ----------------------------------------------------------
        # append memory kv
        # ----------------------------------------------------------

        mk, mv = map(
            lambda t: repeat(
                t,
                'h n d -> bn h n d',
                bn=b * n_blocks
            ),
            self.mem_kv
        )

        k = torch.cat((mk, k), dim=-2)
        v = torch.cat((mv, v), dim=-2)

        # ----------------------------------------------------------
        # local attention
        # ----------------------------------------------------------

        out = self.attend(q, k, v)

        # ----------------------------------------------------------
        # reshape back to windows
        # ----------------------------------------------------------

        out = rearrange(
            out,
            '(b n) h (kh kw) d -> b n (h d) kh kw',
            b=b,
            n=n_blocks,
            kh=kernel_h,
            kw=kernel_w
        )

        # ----------------------------------------------------------
        # crop central valid region
        # remove halo
        # ----------------------------------------------------------

        out = out[
            :,
            :,
            :,
            ph:ph + bh,
            pw:pw + bw
        ]

        # ----------------------------------------------------------
        # stitch blocks back
        # ----------------------------------------------------------

        nH = H // bh
        nW = W // bw

        out = rearrange(
            out,
            'b (nh nw) c bh bw -> b c (nh bh) (nw bw)',
            nh=nH,
            nw=nW
        )

        return self.to_out(out)


class HaloNetFast(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        cond1d_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3, # input dimension
        self_condition = False,
        learned_variance = False,
        attn_dim_head = 32,
        attn_heads = 4,
        attn_modes = None,    # [0 - no attention, 1 - linear, 2 - local, 3 - full]
        flash_attn = True,
        detach_first_resnet_output = False,
        use_linear_layers = True
    ):
        super().__init__()

        # determine dimensions
        self.detach_first_resnet_output = detach_first_resnet_output

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, padding = 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # attention
        if not attn_modes:
            attn_modes = (*((0,) * (len(dim_mults) - 1)), 0)

        num_stages = len(dim_mults)
        attn_modes  = cast_tuple(attn_modes, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(attn_modes) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn, num_mem_kv=8)
        LocalAttention = partial(BlockAttention, num_mem_kv=8, flash=flash_attn, block_size=(8, 8))
        MidAttention = partial(BlockAttention, num_mem_kv=8, flash=flash_attn, block_size=(4, 4))

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), attn_mode, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, attn_modes, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            if attn_mode == 0:
                attn_klass = torch.nn.Identity
            elif attn_mode == 1:
                attn_klass = LinearAttention
            elif attn_mode == 2:
                attn_klass = LocalAttention
            else:
                attn_klass = FullAttention

            self.downs.append(ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = cond1d_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = cond1d_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = cond1d_dim)
        self.mid_attn = MidAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = cond1d_dim)

        for ind, ((dim_in, dim_out), attn_mode, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, attn_modes, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            if attn_mode == 0:
                attn_klass = torch.nn.Identity
            elif attn_mode == 1:
                attn_klass = LinearAttention
            elif attn_mode == 2:
                attn_klass = LocalAttention
            else:
                attn_klass = FullAttention

            self.ups.append(ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = cond1d_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = cond1d_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = cond1d_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, cond1d = None, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, cond1d)
            h.append(x)

            x = block2(x, cond1d)
            # if self.detach_first_resnet_output:
            #     x = attn(x) + x.detach()
            # else:
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, cond1d)
        # if self.detach_first_resnet_output:
        #     x = self.mid_attn(x) + x.detach()
        # else:
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, cond1d)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, cond1d)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, cond1d)
            # if self.detach_first_resnet_output:
            #     x = attn(x) + x.detach()
            # else:
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, cond1d)
        return self.final_conv(x)


def randd(dirs):
    v = torch.randn(dirs, 3)
    return v / ((v**2).sum(dim=-1, keepdim=True) + 0.00000001)


class ScatNet(torch.nn.Module):
    """
    Base class for neural renderers
    """
    extinction_tensor: torch.Tensor = None
    density_scale: float = 10.0
    light_direction: tuple = (0.0, 1.0, 0.0)
    light_intensity: tuple = (1.0, 1.0, 1.0)
    environment_tensor: torch.Tensor = None
    scattering_albedo_tensor: torch.Tensor = None
    g: float = 0.5

    def __init__(self,
                 cameras: int = 1,
                 width: int = 512,
                 height: int = 512,
                 jittered: bool = False,
                 ):
        super().__init__()
        # initialize camera
        self.camera_tensor = rdv.ensure_on_rdv(
            torch.zeros(cameras, 9)
        )
        self.camera_tensor[:,0:3] = randd(cameras) * 3
        self.camera_tensor[:,7] = 1.0 # up vector
        self.sensor = rdv.Cameras(
            width, height,
            poses=self.camera_tensor,
            aspect_ratio=width/height,
            jitter=jittered)
        # initial extinction and majorant
        self.extinction_tensor = rdv.ensure_on_rdv(torch.ones(1,1,1,1), location=rdv.MemoryLocation.DEVICE)
        self.extinction_gpu_sampler = rdv.GPUSampler(self.extinction_tensor, clamp=True, aligned=False, linear=True)
        self.extinction_map = rdv.RefMap(self.extinction_gpu_sampler)
        # coarse majorant for dda
        div = max(1, self.extinction_tensor.shape[1] // 16)
        majorant_grid_shape = [max(1, d//div) for d in self.extinction_tensor.shape[:3]]
        majorant_grid = rdv.TensorGrid(rdv.MajorantGrid(rdv.TensorGrid(self.extinction_tensor), majorant_grid_shape, aligned=False, linear=True)())
        self.majorant_grid = rdv.RefGrid(majorant_grid)
        # sphere-based majorant field
        div = max(1, self.extinction_tensor.shape[1] // 32)
        majorant_grid_shape = [max(1, d//div) for d in self.extinction_tensor.shape[:3]]
        majorant_grid = rdv.TensorGrid(rdv.MajorantGrid(rdv.TensorGrid(self.extinction_tensor), majorant_grid_shape, aligned=False, linear=True)())
        sphere_radius_field_grid = rdv.sphere_radius_field(majorant_grid, levels=16, density_scale=self.density_scale)
        self.majorant_map = rdv.RefMap(rdv.SphereMajorantField(sphere_radius_field_grid))
        self.transform = rdv.trs()

        # initial scattering albedo map
        self.scattering_albedo_tensor = rdv.wrap((0.99, 0.99, 0.99)).as_tensor('torch')
        self.scattering_albedo_map = rdv.Constant(self.scattering_albedo_tensor)

        # initial environment objects
        self.environment_tensor = rdv.ensure_on_rdv(torch.tensor([[[0.1, 0.1, 0.2]]]), location=rdv.MemoryLocation.DEVICE)
        self.environment_gpu_sampler = rdv.GPUSampler(self.environment_tensor, clamp=True, aligned=False, linear=True)
        self.environment_map = rdv.RefMap(rdv.dir2xr >> self.environment_gpu_sampler)
        self.environment_sampler = rdv.XREnvironmentSampler(self.environment_map, self.environment_tensor, 6)
        self.environment_sampler_map = rdv.RefMap(self.environment_sampler)

        # build all maps
        self.build_maps()
        self.maps_are_dirty = False

    def build_maps(self):
        """
        Builds/rebuilds necessary maps.
        """
        self.pbvrt_map = rdv.PBVRT(
            extinction=self.extinction_map,
            majorant=self.majorant_map,
            scattering_albedo=self.scattering_albedo_map,
            environment=self.environment_map,
            light_direction=self.light_direction,
            light_intensity=self.light_intensity,
            g=self.g,
            density_scale=self.density_scale,
            transform=self.transform,
            max_bounces=2048
        )
        self.scattering_capture_grid = self.sensor >> self.pbvrt_map[0:3] >> rdv.average(4)
        self.radiance_capture_grid = self.sensor >> (self.pbvrt_map | rdv.X[3:6]) >> (rdv.X[0:3] + rdv.X[3:4].promote(3) * (rdv.X[4:7] >> self.environment_map)) >> rdv.average(1)
        self.transmittance_map = rdv.RatiotrackingTransmittance(
            extinction=self.extinction_map,
            majorant=self.majorant_map,
            transform=self.transform,
            density_scale=self.density_scale
        )
        self.transmittance_capture_grid = self.sensor >> self.transmittance_map
        self.environment_map_by_direction = rdv.X[3:6] >> self.environment_map
        self.environment_capture_grid = self.sensor >> self.environment_map_by_direction
        self.background_map = self.transmittance_map * self.environment_map_by_direction
        self.background_capture_grid = self.sensor >> self.background_map

        # render_direct_scattering: single-scattering radiance via NEE
        # against the fixed-direction light (no environment involved).
        self.pbvr_nee_map = rdv.PBVR_NEE(
            extinction=self.extinction_map,
            majorant=self.majorant_map,
            scattering_albedo=self.scattering_albedo_map,
            transform=self.transform,
            light_direction=self.light_direction,
            density_scale=self.density_scale,
            g=self.g,
            max_bounces=2048
        )
        self.direct_scattering_capture_grid = self.sensor >> self.pbvr_nee_map >> rdv.average(1)

        # render_environment_scattering: scattering radiance via NEE
        # against the environment itself, importance-sampled through a
        # quadtree over its own solid-angle-weighted luminance (see
        # XREnvironmentSampler) instead of a uniform direction - much
        # lower variance whenever the environment has concentrated bright
        # regions. Rebuilt (not just patched in place) whenever
        # environment_tensor's own content changes - see set_environment -
        # since the quadtree is only ever built once, from a snapshot of
        # environment_tensor, at construction time.
        self.environment_sampler_map = rdv.XREnvironmentSampler(self.environment_map, self.environment_tensor, levels=8)
        self.pbvr_env_map = rdv.PBVR_ENV(
            extinction=self.extinction_map,
            majorant=self.majorant_map,
            scattering_albedo=self.scattering_albedo_map,
            environment_sampler=self.environment_sampler_map,
            transform=self.transform,
            density_scale=self.density_scale,
            g=self.g,
            max_bounces=2048
        )
        self.environment_scattering_capture_grid = self.sensor >> self.pbvr_env_map >> rdv.average(1)

        # render_background: the environment sampled directly along each
        # camera ray's own (fixed) direction - no medium, no importance
        # sampling (there's no direction left to choose - the ray already
        # picked one).
        self.render_background_capture_grid = self.sensor >> self.environment_map_by_direction >> rdv.average(1)

        # render_transmittance: same RatiotrackingTransmittance as
        # transmittance_capture_grid, but averaged over `samples`
        # independent ratio-tracking estimates for lower variance (see
        # render_transmittance/capture_transmittance's own docstrings for
        # why these are kept as two separate grids).
        self.render_transmittance_capture_grid = self.sensor >> self.transmittance_map >> rdv.average(1)

        # capture_transmittance_raymarching: deterministic fixed-step
        # raymarch (RaymarchingTransmittanceTrait) instead of
        # transmittance_map's own majorant-based ratio tracking - no per-
        # dispatch Monte Carlo noise, useful for blending a background pass
        # without needing many samples to average the noise out.
        self.raymarching_transmittance_map = rdv.RaymarchingTransmittance(
            extinction=self.extinction_map,
            transform=self.transform,
            step_size=0.001,
            density_scale=self.density_scale,
        )
        self.transmittance_raymarching_capture_grid = self.sensor >> self.raymarching_transmittance_map

    def update_maps(self):
        """
        Update fields of each built map.
        """
        if not self.maps_are_dirty:
            return
        # update scattering map
        self.pbvrt_map.g = self.g
        self.pbvrt_map.density_scale = self.density_scale
        self.pbvrt_map.transform = rdv.mat4x3(self.transform)
        self.pbvrt_map.light_direction = rdv.vec3(*self.light_direction)
        self.pbvrt_map.light_intensity = self.light_intensity
        # update transmittance map
        self.transmittance_map.transform = rdv.mat4x3(self.transform)
        self.transmittance_map.density_scale = self.density_scale
        self.raymarching_transmittance_map.transform = rdv.mat4x3(self.transform)
        self.raymarching_transmittance_map.density_scale = self.density_scale
        # update direct/environment scattering maps
        self.pbvr_nee_map.g = self.g
        self.pbvr_nee_map.density_scale = self.density_scale
        self.pbvr_nee_map.transform = rdv.mat4x3(self.transform)
        self.pbvr_nee_map.light_direction = rdv.vec3(*self.light_direction)
        self.pbvr_env_map.g = self.g
        self.pbvr_env_map.density_scale = self.density_scale
        self.pbvr_env_map.transform = rdv.mat4x3(self.transform)
        self.maps_are_dirty = False

    def set_environment(self, environment_tensor):
        # Always rebuild (never the in-place .copy_() + .update() path this
        # used to take when the shape matched, which is the common/every-
        # time case - always the same [H, W, 3] loaded from
        # environments.h5). That fast path's torch.cuda.synchronize() +
        # .environment_gpu_sampler.update() combo silently failed to
        # actually refresh the GPU-bound sampler's content - confirmed by
        # forcing this unconditional rebuild and verifying the render
        # actually changes with the new content (see
        # experiments/sanity_check_pbvrt_zero_light.py's own investigation:
        # a "both light and environment zero" control render was coming
        # back with substantial nonzero radiance because the environment
        # update was a no-op, so it was still reading whatever content was
        # uploaded first).
        self.environment_tensor = rdv.ensure_on_rdv(environment_tensor)
        self.environment_gpu_sampler = rdv.GPUSampler(self.environment_tensor, clamp=True, aligned=False, linear=True)
        self.environment_map.map = rdv.dir2xr >> self.environment_gpu_sampler
        # in any case rebuild the importance-sampling quadtree - like
        # set_extinction's own majorant, it depends on the actual
        # environment *content*, not just its shape, so a same-shape
        # in-place .copy_ above needs this refreshed too, or
        # render_environment_scattering would keep importance-sampling
        # the previous content (still unbiased either way - environment_map
        # itself is always read fresh - just needlessly high-variance).
        # pbvr_env_map.environment_sampler was bound (at build_maps time)
        # to self.environment_sampler_map's own RefMap identity/type - so
        # that RefMap object must never be replaced, only its .map content.
        self.environment_sampler_map.map = rdv.XREnvironmentSampler(self.environment_map, self.environment_tensor, levels=8)

    def set_environment_objects(self, env_objects):
        """Takes an already-normalized
        _dataset_common.load_environment_objects(environment_id) result
        (cached per environment_id there, so the disk read + GPUSampler/
        XREnvironmentSampler build cost is paid at most once per
        environment). Swaps the *content* (.map) of this ScatNet's own
        persistent environment_map/environment_sampler_map RefMaps -
        never replaces the RefMap objects themselves, since pbvr_env_map's
        pipeline was compiled against their identity/type at build_maps
        time."""
        self.environment_tensor = env_objects.tensor
        self.environment_gpu_sampler = env_objects.gpu_sampler
        self.environment_map.map = env_objects.map
        self.environment_sampler_map.map = env_objects.environment_sampler
        self.maps_are_dirty = True

    def set_light(self, light_direction: tuple | None = None, light_intensity: tuple | None = None):
        if light_direction is not None:
            self.light_direction = light_direction
        if light_intensity is not None:
            self.light_intensity = light_intensity
        self.maps_are_dirty = True

    def set_scattering_albedo(self, scattering_albedo: tuple):
        self.scattering_albedo_tensor.copy_(torch.tensor(list(scattering_albedo)))

    def set_anisotropy(self, g: float):
        self.g = g
        self.maps_are_dirty = True

    def set_extinction(self, extinction_tensor: torch.Tensor | None = None, density_scale: float | None = None):
        if density_scale is not None:
            self.density_scale = density_scale
            self.maps_are_dirty = True
        if extinction_tensor is not None:
            if self.extinction_tensor.shape == extinction_tensor.shape:
                # only need to update tensor content - but GPUSampler
                # samples a real Vulkan image, uploaded from the tensor
                # ONCE at construction (see its own docstring: "update(t)
                # (re)loads the image from t - never done implicitly") -
                # mutating extinction_tensor's own bytes in place via
                # .copy_ does NOT touch that already-uploaded image, so
                # the GPU-side sampler must be told to re-upload
                # explicitly, or every render after the first keeps
                # reading whatever content was there at construction time.
                # torch.cuda.synchronize() first: .copy_ on the CUDA side
                # is asynchronous, and update()'s own Vulkan-side transfer
                # (a different API, no automatic ordering against pending
                # CUDA work) can otherwise race ahead and read the tensor's
                # still-stale bytes.
                self.extinction_tensor.copy_(extinction_tensor)
                torch.cuda.synchronize()
                self.extinction_gpu_sampler.update()
            else:
                self.extinction_tensor = rdv.ensure_on_rdv(extinction_tensor)
                # update only referred map
                self.extinction_gpu_sampler = rdv.GPUSampler(self.extinction_tensor, clamp=True, aligned=False, linear=True)
                self.extinction_map.map = self.extinction_gpu_sampler
                bmin, bmax = rdv.grid3d_fit_box(self.extinction_tensor.shape[:3], align_corners=False)
                scale, offset = rdv.unit2box(bmin, bmax)
                self.transform = rdv.trs(translation=offset, scale=scale)

        # in any case update majorant
        div = max(1, self.extinction_tensor.shape[1] // 64)
        majorant_grid_shape = [max(1, d // div) for d in self.extinction_tensor.shape[:3]]
        coarse_majorant_grid = rdv.TensorGrid(
            rdv.MajorantGrid(rdv.TensorGrid(self.extinction_tensor), majorant_grid_shape, aligned=False, linear=True)())

        div = max(1, self.extinction_tensor.shape[1] // 32)
        majorant_grid_shape = [max(1, d // div) for d in self.extinction_tensor.shape[:3]]
        majorant_grid = rdv.TensorGrid(
            rdv.MajorantGrid(rdv.TensorGrid(self.extinction_tensor), majorant_grid_shape, aligned=False, linear=True)())
        sphere_radius_field_grid = rdv.sphere_radius_field(majorant_grid, levels=32, density_scale=self.density_scale)
        # update only referred map - RefGrid.shape/._shape mirror whatever
        # it's currently wrapping only at construction time (see its own
        # comment in rdv/maths/__init__.py), so swapping .grid in place
        # like this (instead of rebuilding self.majorant_grid, which would
        # orphan denoiser_features_map's own already-captured reference to
        # this exact RefGrid instance) must refresh them here too, or
        # anything reading the wrapped grid's own dimensions through this
        # RefGrid (e.g. VolumeProfileDDA's DDA traversal) would silently
        # keep using the previous cloud's own majorant_grid_shape forever.
        self.majorant_grid.grid = coarse_majorant_grid
        self.majorant_grid.shape = coarse_majorant_grid.shape
        self.majorant_grid._shape = coarse_majorant_grid.shape
        self.majorant_map.map = rdv.SphereMajorantField(sphere_radius_field_grid)

    def set_camera(self,
        camera_index = 0,
        origin = None,
        target = None,
        up = None,
                   ):
        if origin is not None:
            self.camera_tensor[camera_index][0:3] = torch.tensor(list(origin))
        if target is not None:
            self.camera_tensor[camera_index][3:6] = torch.tensor(list(target))
        if up is not None:
            self.camera_tensor[camera_index][6:9] = torch.tensor(list(up))

    def set_cameras(self, poses: torch.Tensor):
        self.camera_tensor.copy_(poses)

    def capture_radiance(self, samples: int = 1):
        self.update_maps()
        self.radiance_capture_grid.samples = samples
        return self.radiance_capture_grid().as_tensor('torch')

    def capture_scattering(self, samples: int = 1):
        self.update_maps()
        self.scattering_capture_grid.samples = samples
        return self.scattering_capture_grid().as_tensor('torch')

    def capture_transmittance(self):
        """(C, H, W, 1): raymarched transmittance T along each camera ray."""
        self.update_maps()
        return self.transmittance_capture_grid().as_tensor('torch')

    def capture_environment(self):
        """(C, H, W, 3): environment radiance E along each camera ray's own direction (no medium)."""
        self.update_maps()
        return self.environment_capture_grid().as_tensor('torch')

    def capture_background(self):
        """(C, H, W, 3): environment attenuated by the medium's own transmittance (T * E)."""
        self.update_maps()
        return self.background_capture_grid().as_tensor('torch')

    def render_direct_scattering(self, samples: int = 1):
        """(C, H, W, SPECTRAL_DIM): single-scattering radiance via PBVR_NEE
        (next-event estimation against the fixed-direction light), averaged
        over `samples` independent estimates."""
        self.update_maps()
        self.direct_scattering_capture_grid.samples = samples
        return self.direct_scattering_capture_grid().as_tensor('torch')

    def render_environment_scattering(self, samples: int = 1):
        """(C, H, W, SPECTRAL_DIM): scattering radiance via PBVR_ENV (next-
        event estimation against the environment itself), importance-
        sampled through XREnvironmentSampler - see build_maps - averaged
        over `samples` independent estimates."""
        self.update_maps()
        self.environment_scattering_capture_grid.samples = samples
        return self.environment_scattering_capture_grid().as_tensor('torch')

    def render_background(self, samples: int = 1):
        """(C, H, W, 3): environment radiance sampled directly along each
        camera ray's own direction - no medium, no importance sampling
        (the ray's direction is already fixed, nothing left to choose)."""
        self.update_maps()
        self.render_background_capture_grid.samples = samples
        return self.render_background_capture_grid().as_tensor('torch')

    def render_transmittance(self, samples: int = 1):
        """(C, H, W, 1): RatiotrackingTransmittance T along each camera ray,
        averaged over `samples` independent ratio-tracking estimates."""
        self.update_maps()
        self.render_transmittance_capture_grid.samples = samples
        return self.render_transmittance_capture_grid().as_tensor('torch')

    def capture_transmittance_raymarching(self):
        """(C, H, W, 1): deterministic fixed-step raymarched transmittance T
        along each camera ray - no Monte Carlo noise (unlike
        capture_transmittance's own RatiotrackingTransmittance), so a single
        call is already a clean result, useful for blending a background
        term without needing to average away noise."""
        self.update_maps()
        return self.transmittance_raymarching_capture_grid().as_tensor('torch')

    def render_full_radiance(self, samples: int = 1):
        """(C, H, W, SPECTRAL_DIM): direct_scattering*light_intensity +
        environment_scale*(environment_scattering + transmittance*background) -
        light_intensity/environment_scale are post-hoc intensity knobs on top
        of the same physically-based renders (render_direct_scattering/
        render_environment_scattering/render_transmittance/render_background),
        not part of the scene itself."""
        self.update_maps()
        self.radiance_capture_grid.samples = samples
        return self.radiance_capture_grid().as_tensor('torch')


class DenoiserScatNet(ScatNet):
    """
    Small ndvpt_modeling.Autoencoder that denoises a noisy PBVRT render
    (PBVRTPort - see pbvrt_port.h, an exact port of rendervous.project's own
    historical pbvrt.h, NOT the current framework's differently-formulated
    rdv.PBVRT - see that class's own docstring), guided by a cheap "first
    scattering guess" feature buffer (FSGProfile - see fsg_profile.h, an
    exact port of old_profiles/profile_fsg.h). Both ports were needed to
    exactly reproduce the Denoiser checkpoint's own training-time channel
    layout (confirmed against rendervous.project/neural_rendering's own
    train_denoiser.py) - reusing the modern framework's own PBVRT/whatever
    replaced ProfileFSG would silently feed the model differently-computed
    channels despite matching shapes. Same structural pattern as
    HaloScatNet2: ScatNet is used purely as a sample provider (self.sensor
    >> profiler) - this class owns no tracing logic of its own.
    condition = PBVRTPort(4ch: rgb + still-background flag) | FSGProfile
    (5ch: depth, albedo, g) = 9ch - exact order/channels train_denoiser.py
    trained on. FSGProfile's own trailing S channel (binary hit indicator,
    not part of the 9ch model input) is used only for blending - see
    capture_reconstructed_radiance.
    """
    def __init__(self, cameras: int = 1, width: int = 512, height: int = 512, jittered: bool = False):
        super().__init__(cameras, width, height, jittered)
        from . import ndvpt_modeling
        self.model = ndvpt_modeling.Autoencoder(input_channels=9, output_channels=3)

    def build_maps(self):
        super().build_maps()
        from ._profiles import FSGProfile, PBVRTPort
        self.profiler = FSGProfile(
            extinction=self.extinction_map, majorant=self.majorant_map, majorant_grid=self.majorant_grid,
            scattering_albedo=self.scattering_albedo_map, density_scale=self.density_scale,
            transform=self.transform, g=self.g,
        )
        self.noisy_radiance_map = PBVRTPort(
            extinction=self.extinction_map, majorant=self.majorant_map, environment=self.environment_map,
            scattering_albedo=self.scattering_albedo_map, transform=self.transform,
            light_direction=self.light_direction, light_intensity=self.light_intensity,
            density_scale=self.density_scale, g=self.g,
        )
        # single combined dispatch - noisy radiance (4ch, averaged x4
        # internally) | profile features (5ch) | background (3ch,
        # environment along the ray) - replaces 3 separate captures with
        # one, same optimization as HaloScatNet2's own
        # profile_capture_with_background. No environment_scale rescale
        # needed on the background term here (unlike Halo's) - this class's
        # own environment is never normalized, kept at true physical scale
        # throughout (see class docstring).
        self.combined_map = (self.noisy_radiance_map >> rdv.average(4)) | self.profiler \
            | (rdv.X[3:6] >> self.environment_map)
        self.combined_capture = self.sensor >> self.combined_map

    def update_maps(self):
        if not self.maps_are_dirty:
            return
        self.profiler.g = self.g
        self.profiler.density_scale = self.density_scale
        self.profiler.transform = rdv.mat4x3(self.transform)
        self.noisy_radiance_map.g = self.g
        self.noisy_radiance_map.density_scale = self.density_scale
        self.noisy_radiance_map.transform = rdv.mat4x3(self.transform)
        self.noisy_radiance_map.light_direction = rdv.vec3(*self.light_direction)
        self.noisy_radiance_map.light_intensity = rdv.vec3(*self.light_intensity)
        super().update_maps()

    def load_pretrained(self, model_state):
        """model_state: a state_dict, a path to one (saved under
        pretrained/ in this repo), or a full training checkpoint dict with
        a 'model' key."""
        if isinstance(model_state, (str, bytes)) or hasattr(model_state, "__fspath__"):
            model_state = torch.load(model_state, map_location='cuda')
        if isinstance(model_state, dict) and 'model' in model_state and not any(
                torch.is_tensor(v) for v in model_state.values()):
            model_state = model_state['model']
        self.model.load_state_dict(model_state)
        torch._dynamo.config.cache_size_limit = 128
        self.model = torch.compile(self.model.cuda().half().to(memory_format=torch.channels_last).eval())

    def capture_condition(self) -> tuple:
        """(1,H,W,9) model input, (1,H,W,3) background, (1,H,W,1) S - for
        blending, uses PBVRTPort's own trailing alpha/transmittance channel
        (c[...,3:4], already part of model_input's own first 4 channels),
        NOT FSGProfile's own trailing hit flag (f_full[...,5:6]): PBVRTPort
        is dispatched averaged over 4 samples (see build_maps's own
        combined_map), so its alpha channel is a smoothed 4-sample MC
        transmittance estimate, while FSGProfile's is a single noisy 0/1
        delta-tracking sample (see fsg_profile.h's own docstring on why -
        it never computes a full-ray tau, so a deterministic 1-exp(-tau)
        isn't available there) - using the harder, noisier FSGProfile flag
        for blending produced visibly worse edge merging than DenoiserX's
        own ProfileFDTNEX7Port S (which IS a smooth deterministic
        1-exp(-tau)) despite both nominally doing the same radiance+
        background*(1-S) blend."""
        self.update_maps()
        c = self.combined_capture().as_tensor('torch')
        f_full = c[..., 4:10]  # FSGProfile: 6ch (depth, albedo x3, g, trailing S - unused now)
        model_input = torch.cat([c[..., 0:4], f_full[..., 0:5]], dim=-1)
        # PBVRTPort's own channel is 1.0 while STILL background (not yet
        # scattered), 0.0 once resolved - the inverse of what the
        # radiance + background*(1-S) blend below needs (S should be ~1
        # where the model's own radiance should dominate, ~0 where raw
        # background should show through) - flip it.
        S = 1.0 - c[..., 3:4]
        background = c[..., 10:13]
        return model_input, background, S

    @torch.no_grad()
    def capture_reconstructed_scattering(self):
        dtype = next(self.model.parameters()).dtype
        condition, _, _ = self.capture_condition()
        cond = condition.permute(0, 3, 1, 2).to(dtype).contiguous(memory_format=torch.channels_last)
        radiance = self.model(cond)
        return radiance.permute(0, 2, 3, 1).float()

    @torch.no_grad()
    def capture_reconstructed_radiance(self):
        dtype = next(self.model.parameters()).dtype
        condition, background, S = self.capture_condition()
        cond = condition.permute(0, 3, 1, 2).to(dtype).contiguous(memory_format=torch.channels_last)
        radiance = self.model(cond).permute(0, 2, 3, 1).float()
        return radiance + background * (1 - S)


class DenoiserXScatNet(ScatNet):
    """
    Fuller ndvpt_modeling.Autoencoder counterpart to DenoiserScatNet: same
    "denoise a noisy PBVRTPort render (see DenoiserScatNet's own docstring),
    guided by an auxiliary feature buffer" design, but using
    VolumeDeepProfileNoNEE (see volume_deep_profile_no_nee.h) - HaloRT's own
    depth/tr/g/sa feature engineering (volume_deep_profile.h) with the NEE/
    environment-dome blocks stripped out, in place of the old byte-faithful
    ProfileFDTNEX7Port port (profile_fdtnex7_port.h) this class used before
    - this architecture is retrained from scratch (see src/training/
    _denoiser_channel_training.py), so there's no old-checkpoint feature-
    layout compatibility to preserve anymore. Same ScatNet-as-sample-provider
    pattern as HaloScatNet2/DenoiserScatNet.
    condition = PBVRTPort(4ch: rgb + deterministic transmittance exp(-tau),
    NOT PBVRTPort's own noisy "still background" flag - see
    capture_condition's own note) | VolumeDeepProfileNoNEE's own [0..8]
    (9ch: depth, tr, g, sa - no NEE/env-dome at all, see
    volume_deep_profile_no_nee.h's own header comment) = 13ch.
    VolumeDeepProfileNoNEE's own trailing S channel (not part of the 13ch
    model input) is used only for blending - see
    capture_reconstructed_radiance.
    """
    TRANSMITTANCE_THRESHOLDS = [4.0, 1.0, 0.25, 0.125]  # same crossing thresholds as HaloRT's own VolumeDeepProfile
    GUIDE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # everything but the trailing S

    def __init__(self, cameras: int = 1, width: int = 512, height: int = 512, jittered: bool = False):
        super().__init__(cameras, width, height, jittered)
        from . import ndvpt_modeling
        self.model = ndvpt_modeling.Autoencoder(input_channels=13, output_channels=3)

    def build_maps(self):
        super().build_maps()
        from ._profiles import VolumeDeepProfileNoNEE, PBVRTPort
        self.profiler = VolumeDeepProfileNoNEE(
            extinction=self.extinction_map, majorant=self.majorant_map, majorant_grid=self.majorant_grid,
            scattering_albedo=self.scattering_albedo_map,
            transmittance_threshold=self.TRANSMITTANCE_THRESHOLDS, density_scale=self.density_scale,
            transform=self.transform, g=self.g, step_size=0.001, sample_index=0,
            use_depth_encoding=True, use_reduced_transmittance=True,
            use_similarity_anisotropy=False, use_scattering_accumulation=False,
        )
        self.noisy_radiance_map = PBVRTPort(
            extinction=self.extinction_map, majorant=self.majorant_map, environment=self.environment_map,
            scattering_albedo=self.scattering_albedo_map, transform=self.transform,
            light_direction=self.light_direction, light_intensity=self.light_intensity,
            density_scale=self.density_scale, g=self.g,
        )
        # single combined dispatch - noisy radiance (4ch, averaged x4
        # internally) | full profile (10ch, sliced down to GUIDE_IDX after
        # capture) | background (3ch) - see DenoiserScatNet's own identical
        # comment for the rationale (no environment_scale rescale needed
        # either, same reason).
        self.combined_map = (self.noisy_radiance_map >> rdv.average(4)) | self.profiler \
            | (rdv.X[3:6] >> self.environment_map)
        self.combined_capture = self.sensor >> self.combined_map

    def update_maps(self):
        if not self.maps_are_dirty:
            return
        self.profiler.g = self.g
        self.profiler.density_scale = self.density_scale
        self.profiler.transform = rdv.mat4x3(self.transform)
        self.noisy_radiance_map.g = self.g
        self.noisy_radiance_map.density_scale = self.density_scale
        self.noisy_radiance_map.transform = rdv.mat4x3(self.transform)
        self.noisy_radiance_map.light_direction = rdv.vec3(*self.light_direction)
        self.noisy_radiance_map.light_intensity = rdv.vec3(*self.light_intensity)
        super().update_maps()

    def load_pretrained(self, model_state):
        """model_state: a state_dict, a path to one (saved under
        pretrained/ in this repo), or a full training checkpoint dict with
        a 'model' key."""
        if isinstance(model_state, (str, bytes)) or hasattr(model_state, "__fspath__"):
            model_state = torch.load(model_state, map_location='cuda')
        if isinstance(model_state, dict) and 'model' in model_state and not any(
                torch.is_tensor(v) for v in model_state.values()):
            model_state = model_state['model']
        self.model.load_state_dict(model_state)
        torch._dynamo.config.cache_size_limit = 128
        self.model = torch.compile(self.model.cuda().half().to(memory_format=torch.channels_last).eval())

    def capture_condition(self) -> tuple:
        """(1,H,W,13) model input, (1,H,W,3) background, (1,H,W,1) S
        (deterministic interaction probability, for blending only) - all
        from the same single combined dispatch (see build_maps).

        Channel 3 is the deterministic transmittance exp(-tau) = 1-S,
        NOT PBVRTPort's own trailing "still background" flag (a noisy
        single-sample 0/1 indicator of whether THIS particular random walk
        happened to scatter) - replacing it with the smooth, low-variance
        tau-based quantity gives the model a reliable depth/occlusion cue
        at this channel instead of per-sample noise."""
        self.update_maps()
        c = self.combined_capture().as_tensor('torch')
        f_full = c[..., 4:14]  # VolumeDeepProfileNoNEE: 10ch (9 guide channels + trailing S)
        S = f_full[..., 9:10]
        deterministic_transmittance = 1.0 - S  # exp(-tau)
        model_input = torch.cat([c[..., 0:3], deterministic_transmittance, f_full[..., self.GUIDE_IDX]], dim=-1)
        background = c[..., 14:17]
        return model_input, background, S

    @torch.no_grad()
    def capture_reconstructed_scattering(self):
        dtype = next(self.model.parameters()).dtype
        condition, _, _ = self.capture_condition()
        cond = condition.permute(0, 3, 1, 2).to(dtype).contiguous(memory_format=torch.channels_last)
        radiance = self.model(cond)
        return radiance.permute(0, 2, 3, 1).float()

    @torch.no_grad()
    def capture_reconstructed_radiance(self):
        dtype = next(self.model.parameters()).dtype
        condition, background, S = self.capture_condition()
        cond = condition.permute(0, 3, 1, 2).to(dtype).contiguous(memory_format=torch.channels_last)
        radiance = self.model(cond).permute(0, 2, 3, 1).float()
        return radiance + background * (1 - S)


def normalize_environment(environment: torch.Tensor):
    """environment / environment.mean() (per channel), so ScatNet always
    renders against a unit-mean environment - `scale` (the per-channel
    mean) is saved alongside so the original can be recovered later as
    `normalized * scale`. A degenerate (non-positive-mean) channel would
    make that division blow up, so that case is substituted with a flat
    white environment and scale=0 (reconstruction is then always 0
    regardless of the substituted normalized content)."""
    scale = environment.mean(dim=(0, 1))  # (3,)
    eps = 0.000001 if scale.min() <= 0.0 else 0.0
    return environment / (scale + eps), scale


class HaloRTCompose(Module):
    """Wraps a HaloScatNet2's per-channel dir_model/env_model UNets
    together with the channel-gather/cast, concat/permute, and final
    light*direct + environment_scale*(environment*S + B*(1-S)) blend that
    otherwise runs as eager-mode glue between 6 separate per-channel model
    calls (see infer_directional_scattering/infer_environment_scattering/
    reconstruct_radiance) - wrapping the WHOLE thing in one torch.compile()
    (see HaloScatNet2.load_pretrained) lets inductor schedule/fuse the
    gather, concat, permute and blend elementwise ops together with (and
    around) both UNets' own kernels, instead of leaving them as separate
    eager kernel launches with Python dispatch overhead stitching together
    6 independently-compiled model calls.

    dir_model/env_model must be RAW (not individually torch.compile'd)
    HaloNetFast instances of their own, not HaloScatNet2's own (already
    individually compiled) self.dir_model/self.env_model - nesting an
    already-compiled submodule inside another compiled module just adds a
    graph-break boundary at that call instead of letting inductor fuse
    across it, so this needs its own raw copies (same weights, see
    load_pretrained) to get the fusion the user actually wants."""

    def __init__(self, dir_model, env_model, dir_indices, env_indices):
        super().__init__()
        self.dir_model = dir_model
        self.env_model = env_model
        self.dir_indices = [list(idxs) for idxs in dir_indices]
        self.env_indices = [list(idxs) for idxs in env_indices]

    def forward(self, condition, background, S, light_intensity, environment_scale):
        """condition: (C,17,H,W) channels-first, as returned by
        capture_condition()[0]. background: (C,H,W,3). S: (C,H,W,1).
        light_intensity: (3,) tensor. environment_scale: scalar/(3,) tensor.
        Returns (C,H,W,3) final radiance - same math as reconstruct_
        radiance()'s own 3-step (infer_directional_scattering +
        infer_environment_scattering + blend) computation."""
        direct = torch.cat([
            self.dir_model(condition[:, idxs].half().contiguous(memory_format=torch.channels_last))
            for idxs in self.dir_indices
        ], dim=1).permute(0, 2, 3, 1)
        environment = torch.cat([
            self.env_model(condition[:, idxs].half().contiguous(memory_format=torch.channels_last))
            for idxs in self.env_indices
        ], dim=1).permute(0, 2, 3, 1)
        return (direct * light_intensity + environment_scale * environment) * S \
            + background * environment_scale * (1 - S)


class HaloScatNet2(ScatNet):
    """
    Spectral-independent per-channel reconstruction (see training/
    _halo_channel_training.py and experiments/paper_teaser_A.py): two
    separate HaloNetFast models (the same weights evaluated once per color
    channel, 8 input features -> 1 output), one trained on the direct-
    scattering target (mode "dir", conditioned on the noisy NEE feature)
    and one on the environment-scattering target (mode "env", conditioned
    on that channel's own noisy environment sample).

    VolumeDeepProfile output layout (DEPTH=4, OUT=14): 0..3 depth samples |
    4 tr | 5 g | 6 nee | 7..9 env rgb | 10..12 scattering albedo rgb |
    13 interaction probability S.
    Per-channel model input (8): depth(4), tr, g, <nee | env_c>, sa_c -
    output is multiplied by S.
    """
    TRANSMITTANCE_THRESHOLDS = [4.0, 1.0, 0.25, 0.125]
    # DEPTH = len(TRANSMITTANCE_THRESHOLDS)
    # STATIC_IDX = list(range(DEPTH + 2)) + [DEPTH + 6, DEPTH + 7, DEPTH + 8, DEPTH + 9]
    # NOISY_IDX = {'dir': [DEPTH + 2], 'env': [DEPTH + 3, DEPTH + 4, DEPTH + 5]}
    # STATIC_SA0 = DEPTH + 2
    # STATIC_S = DEPTH + 5
    #
    INDICES = {
        'dir': [
            [0, 1, 2, 3, 4, 5, 6, 10],  # depth, d1, d2, d3, tau, g, nee, sa_r
            [0, 1, 2, 3, 4, 5, 6, 11],  # depth, d1, d2, d3, tau, g, nee, sa_g
            [0, 1, 2, 3, 4, 5, 6, 12],  # depth, d1, d2, d3, tau, g, nee, sa_b
        ],
        'env': [
            [0, 1, 2, 3, 4, 5, 7, 10],  # depth, d1, d2, d3, tau, g, env_r, sa_r
            [0, 1, 2, 3, 4, 5, 8, 11],  # depth, d1, d2, d3, tau, g, env_g, sa_g
            [0, 1, 2, 3, 4, 5, 9, 12],  # depth, d1, d2, d3, tau, g, env_b, sa_b
        ]
    }
    environment_samples: int = 16  # class-level default: ScatNet.__init__ calls
    # this class's own build_maps() (needing environment_samples) before an
    # nn.Module subclass can assign any instance attribute of its own - see
    # __init__, which corrects this to the constructor argument afterward.
    USE_SIMILARITY_ANISOTROPY = False  # matches the production checkpoints'
    # own training flag (--no-use-similarity-anisotropy was passed, see
    # training/_halo_channel_training.py) - subclasses whose own training
    # run did NOT disable this flag (e.g. HaloScatNetNoAttn, HaloScatNet3 -
    # both launched with "all on", no --no-use-* flags) MUST override this
    # to True, or build_maps()'s VolumeDeepProfile computes tr/depth
    # features differently than what that checkpoint was actually trained
    # on - a real train/inference mismatch, not just a cosmetic flag (see
    # deep_halo_viewer.py's own PROFILE_BY_ARCH fix for the same bug).

    def __init__(self, cameras: int = 1, width: int = 512, height: int = 512, jittered: bool = False,
                 environment_samples: int = 16):
        super().__init__(cameras, width, height, jittered)
        self.environment_samples = environment_samples
        self.maps_are_dirty = True  # re-sync profiler.environment_samples (see update_maps)
        # to the constructor argument on the next update_maps() call, since
        # the class default above was what build_maps() actually just used.
        self.dir_model = self._build_model()
        self.env_model = self._build_model()

    def set_environment(self, environment_tensor):
        environment_tensor, environment_scale = normalize_environment(environment_tensor)
        self.environment_scale = environment_scale.cuda()
        super().set_environment(environment_tensor)
        
    def set_light(self, light_direction: tuple | None = None, light_intensity: tuple | None = None):
        if light_intensity is not None:
            self.light_intensity_tensor = torch.as_tensor([*light_intensity], device='cuda')
        super().set_light(light_direction, light_intensity)

    @staticmethod
    def _build_model():
        return HaloNetFast(
            channels=8, dim=16, out_dim=1, attn_heads=4, attn_dim_head=4,
            dim_mults=(1, 1, 1, 2, 2, 4, 4), attn_modes=(0, 0, 0, 2, 0, 2, 0),
            use_linear_layers=False,
        )

    def build_maps(self):
        super().build_maps()
        # deferred import: _profiles is loaded after _modeling by rdv_nr's
        # own __init__ (`from ._modeling import *` then `from ._profiles
        # import *`), so a top-level import here would be circular.
        from ._profiles import VolumeDeepProfile
        self.profiler = VolumeDeepProfile(
            extinction=self.extinction_map, majorant=self.majorant_map, majorant_grid=self.majorant_grid,
            density_scale=self.density_scale, environment_sampler=self.environment_sampler_map,
            light_direction=self.light_direction, step_size=0.001, transform=self.transform, sample_index=0,
            transmittance_threshold=self.TRANSMITTANCE_THRESHOLDS, scattering_albedo=self.scattering_albedo_map,
            g=self.g, environment_samples=self.environment_samples, use_depth_encoding=True,
            use_reduced_transmittance=True, use_similarity_anisotropy=self.USE_SIMILARITY_ANISOTROPY,
            use_scattering_accumulation=True,
        )
        self.profile_capture = self.sensor >> self.profiler
        self.profile_capture_with_background = self.sensor >> (self.profiler | (rdv.X[3:6] >> self.environment_map))

    def update_maps(self):
        if not self.maps_are_dirty:
            return
        self.profiler.g = self.g
        self.profiler.density_scale = self.density_scale
        self.profiler.transform = rdv.mat4x3(self.transform)
        self.profiler.light_direction = rdv.vec3(*self.light_direction)
        self.profiler.environment_samples = self.environment_samples
        super().update_maps()

    @staticmethod
    def _load_state_dict_file(path):
        """Accepts either a bare state_dict (the format saved under
        pretrained/ in this repo) or a full training checkpoint dict with
        a 'model' key (the original rdv_nr training scripts' own format)."""
        state = torch.load(path, map_location='cuda')
        if isinstance(state, dict) and 'model' in state and not any(
                torch.is_tensor(v) for v in state.values()):
            state = state['model']
        return state

    def load_pretrained(self, dir_path: str | None = None, env_path: str | None = None):
        """dir_path/env_path: either can be omitted to leave that model at
        its current (e.g. already-loaded) weights - see
        _load_state_dict_file for the accepted checkpoint formats."""
        if dir_path is not None:
            state = self._load_state_dict_file(dir_path)
            self.dir_model.load_state_dict(unwrap_model_parameters(
                {k.removeprefix("_orig_mod."): v for k, v in state.items()}))
        if env_path is not None:
            state = self._load_state_dict_file(env_path)
            self.env_model.load_state_dict(unwrap_model_parameters(
                {k.removeprefix("_orig_mod."): v for k, v in state.items()}))
        torch._dynamo.config.cache_size_limit = 128
        self.dir_model = torch.compile(self.dir_model.cuda().half().to(memory_format=torch.channels_last).eval())
        self.env_model = torch.compile(self.env_model.cuda().half().to(memory_format=torch.channels_last).eval())

        # Separate raw (uncompiled) UNet instances with the same weights,
        # dedicated to radiance_compose - see HaloRTCompose's own docstring
        # for why these must be their own instances rather than reusing
        # self.dir_model/self.env_model above (already individually
        # torch.compile'd for infer_directional_scattering/infer_
        # environment_scattering's own call sites). Duplicating a dim=16
        # UNet's worth of half-precision weights is cheap; the point is
        # giving the WHOLE composition - per-channel gather/cast, both
        # UNets, the concat/permute, and the final light/environment/
        # background blend - exactly one torch.compile() of its own, so
        # inductor can fuse/schedule the surrounding glue instead of
        # leaving it as eager-mode ops between 6 separately-compiled calls.
        raw_dir_model = self._build_model().cuda().half().to(memory_format=torch.channels_last).eval()
        raw_env_model = self._build_model().cuda().half().to(memory_format=torch.channels_last).eval()
        # self.dir_model/self.env_model are torch.compile'd OptimizedModules
        # here - their own state_dict() keys carry an "_orig_mod." prefix.
        raw_dir_model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in self.dir_model.state_dict().items()})
        raw_env_model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in self.env_model.state_dict().items()})
        self.radiance_compose = torch.compile(HaloRTCompose(
            raw_dir_model, raw_env_model, self.INDICES['dir'], self.INDICES['env']
        ).cuda())

    def capture_condition(self) -> tuple:
        """(C, 14, H, W): channels-first VolumeDeepProfile features, ready
        to be gathered/sliced per-channel by infer_directional_scattering/
        infer_environment_scattering.
        (C, H, W, 1): channels-last interaction probability computed
        """
        self.update_maps()
        c = self.profile_capture_with_background().as_tensor('torch')
        return c.permute(0, 3, 1, 2), c[..., 13:14], c[..., 14:17]

    @torch.no_grad()
    def infer_directional_scattering(self, condition):
        """condition: this ScatNet's own capture_condition() output.
        (C, 3, H, W): direct-scattering radiance, NOT yet scaled by the
        light's own intensity - see reconstruct_radiance, matching
        render_direct_scattering's identical convention."""
        indices = self.INDICES['dir']
        return torch.cat([
            self.dir_model(condition[:, idxs].half().contiguous(memory_format=torch.channels_last))
            for idxs in indices
        ], dim=1).permute(0, 2, 3, 1)

    @torch.no_grad()
    def infer_environment_scattering(self, condition):
        """condition: this ScatNet's own capture_condition() output.
        (C, H, W, 3): environment-scattering radiance, matching
        render_environment_scattering's own convention (already at the
        environment's own scale - see reconstruct_radiance)."""
        indices = self.INDICES['env']
        return torch.cat([
            self.env_model(condition[:, idxs].half().contiguous(memory_format=torch.channels_last))
            for idxs in indices
        ], dim=1).permute(0, 2, 3, 1)

    @torch.no_grad()
    def infer_directional_scattering_single_channel(self, condition):
        """Like infer_directional_scattering, but evaluates dir_model only
        ONCE (using the 'r' channel's own feature indices) and broadcasts
        that single result to all 3 output channels - valid (and 3x
        cheaper) whenever scattering_albedo is uniform across channels,
        since all 3 channels' inputs would otherwise be identical anyway
        (they differ only in the trailing scattering-albedo feature index -
        see INDICES['dir']). (C, H, W, 3)."""
        idxs = self.INDICES['dir'][0]
        out = self.dir_model(condition[:, idxs].half().contiguous(memory_format=torch.channels_last))
        return out.permute(0, 2, 3, 1).expand(-1, -1, -1, 3)

    @torch.no_grad()
    def reconstruct_direct_only_single_channel(self):
        """Direct-light-only reconstruction (no environment/background
        term, one model evaluation instead of three) - for controlled
        comparisons against renderers with no environment input at all
        (e.g. MRPNN rendered with a black HDRI and only --lightcolor set)."""
        condition, S, B = self.capture_condition()
        direct = self.infer_directional_scattering_single_channel(condition)
        return direct * self.light_intensity_tensor * S

    @torch.no_grad()
    def reconstruct_radiance(self):
        """(C, H, W, 3): the learned drop-in equivalent of render_full_radiance
        - direct_scattering*light_intensity + environment_scale*(environment_
        scattering + transmittance*background), using this model's own
        infer_directional_scattering/infer_environment_scattering in place
        of true Monte Carlo NEE, but the same (still physically simulated)
        capture_transmittance/capture_background for the background term."""
        condition, S, B = self.capture_condition()
        direct = self.infer_directional_scattering(condition)
        environment = self.infer_environment_scattering(condition)
        return (direct * self.light_intensity_tensor + self.environment_scale * environment) * S \
            + B * self.environment_scale * (1 - S)

    @torch.no_grad()
    def reconstruct_radiance_fused(self):
        """Same output as reconstruct_radiance(), but evaluates radiance_
        compose (see HaloRTCompose, built in load_pretrained) instead of
        separately calling infer_directional_scattering/infer_environment_
        scattering and blending eagerly - the per-channel gather/cast, both
        UNets, the concat/permute and the final blend all live inside one
        torch.compile()'d module instead of 6 independently-compiled model
        calls stitched together by eager glue ops."""
        condition, S, B = self.capture_condition()
        return self.radiance_compose(condition, B, S, self.light_intensity_tensor, self.environment_scale)
