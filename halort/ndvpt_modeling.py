"""
Copied from: Neural Denoising for Path Tracing of Medical Volumetric Data
https://github.com/nihofm/ndptmvd/blob/master/src/models.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_ssim_measure = None

def ssim(img1, img2):
    global _ssim_measure
    if _ssim_measure is None:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        _ssim_measure = StructuralSimilarityIndexMeasure(data_range=1.0)
    return _ssim_measure(img1, img2)

# -----------------------------------------------------------
# autoencoder modules

class Encoder(nn.Module):
    def __init__(self, f_in, f_out):
        super(Encoder, self).__init__()
        self.add_module('conv', nn.Conv2d(f_in, f_out, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        skip = x
        x = F.leaky_relu(self.conv(x), negative_slope=0.1)
        return F.max_pool2d(x, kernel_size=2), skip

class Decoder(nn.Module):
    def __init__(self, f_in, f_out):
        super(Decoder, self).__init__()
        self.add_module('conv1', nn.ConvTranspose2d(f_in, f_out, kernel_size=3, stride=1, padding=1))
        self.add_module('conv2', nn.ConvTranspose2d(f_out, f_out, kernel_size=3, stride=1, padding=1))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.size()[-2:])
        x = torch.cat((x, skip), dim=-3)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_channels, output_channels=3, stages=5, hidden=48):
        super(Autoencoder, self).__init__()
        stages = max(2, stages) # ensure min 2 stages
        # input layer
        self.add_module("input", nn.Conv2d(input_channels, hidden, kernel_size=3, stride=1, padding=1))
        # encoder modules
        self.encoders = nn.ModuleList()
        for i in range(stages):
            self.encoders.append(Encoder(hidden, hidden))
        # latent space
        self.add_module("latent", nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1))
        # decoder modules
        self.decoders = nn.ModuleList()
        for i in range(stages):
            if i == 0: # first decoder: hidden + skip_connections feature maps
                self.decoders.append(Decoder(2*hidden, 2*hidden))
            else: # intermediate decoder: 2*hidden + skip_connections feature maps
                self.decoders.append(Decoder(3*hidden, 2*hidden))
        # output layer
        self.add_module('dropout', nn.Dropout2d(0.1))
        self.add_module("output", nn.ConvTranspose2d(2*hidden, output_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.input(x * 2 - 1) # zero centered input
        # encoder stages
        skip_connections = []
        for i, encoder in enumerate(self.encoders):
            x, skip = encoder(x)
            skip_connections.append(skip)
        # latent space
        x = self.latent(x)
        # decoder stages
        skip_connections.reverse()
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        x = self.dropout(x)
        return self.output(x) * 0.5 + 0.5 # output in [0, 1]

class AutoencoderDualF24(nn.Module):
    def __init__(self):
        super(AutoencoderDualF24, self).__init__()
        self.add_module("aec_direct", Autoencoder(15, 3, stages=5, hidden=48))
        self.add_module("aec_indirect", Autoencoder(15, 3, stages=5, hidden=48))
        self.add_module("output", nn.ConvTranspose2d(6, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        direct = self.aec_direct(x[..., 0:15, :, :])
        indirect = self.aec_indirect(torch.cat((x[..., 0:3, :, :], direct, x[..., 15:24, :, :]), dim=-3))
        x = self.output(torch.cat((direct * 2 - 1, indirect * 2 - 1), dim=-3)) * 0.5 + 0.5
        return x

class AutoencoderDualF24Big(nn.Module):
    def __init__(self):
        super(AutoencoderDualF24Big, self).__init__()
        self.add_module("aec_direct", Autoencoder(15, 12, stages=5, hidden=48))
        self.add_module("aec_indirect", Autoencoder(24, 12, stages=5, hidden=48))
        self.add_module("output", nn.ConvTranspose2d(24, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        direct = self.aec_direct(x[..., 0:15, :, :])
        indirect = self.aec_indirect(torch.cat((x[..., 0:3, :, :], direct, x[..., 15:24, :, :]), dim=-3))
        x = self.output(torch.cat((direct * 2 - 1, indirect * 2 - 1), dim=-3)) * 0.5 + 0.5
        return x


def psnr(img1, img2):
    return 20 * torch.log10(torch.tensor(1, dtype=torch.float32, device=img1.device)) - 10 * torch.log10(F.mse_loss(img1, img2))

def spec_mask(x):
    size = 11 # px
    kernel = torch.ones((3, 1, size, size)) / (size**2)
    if x.is_cuda: kernel = kernel.cuda()
    loc_mean = torch.squeeze(F.conv2d(torch.unsqueeze(x, dim=0), kernel, padding=size//2, groups=3), dim=0)
    return torch.clamp(x - loc_mean, 0, 1)



class L1HFENLoss(torch.nn.Module):
    def __init__(self, channels = 3):
        super(L1HFENLoss, self).__init__()
        self.channels = channels
        self.kernel = torch.Tensor([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]]).repeat((self.channels, 1, 1, 1))

    def forward(self, img1, img2):
        # L1 loss
        L1 = F.l1_loss(img1, img2)
        # HFEN loss
        if img1.is_cuda: self.kernel = self.kernel.cuda()
        log1 = F.conv2d(img1, self.kernel, padding=2, groups=self.channels)
        log2 = F.conv2d(img2, self.kernel, padding=2, groups=self.channels)
        Lhf = F.l1_loss(log1, log2)
        return 0.8 * L1 + 0.2 * Lhf

class L1HFENSpecLoss(L1HFENLoss):
    def __init__(self, channels = 3):
        super(L1HFENSpecLoss, self).__init__(channels)
        self.channels = channels
        self.kernel_mean = torch.ones((channels, 1, 11, 11)) / (11**2)

    def forward(self, img1, img2):
        # L1 loss
        L1 = F.l1_loss(img1, img2)
        # HFEN loss
        if img1.is_cuda: self.kernel = self.kernel.cuda()
        log1 = F.conv2d(img1, self.kernel, padding=2, groups=self.channels)
        log2 = F.conv2d(img2, self.kernel, padding=2, groups=self.channels)
        Lhf = F.l1_loss(log1, log2)
        # spec loss
        if img1.is_cuda: self.kernel_mean = self.kernel_mean.cuda()
        lm1 = F.conv2d(img1, self.kernel_mean, padding=11//2, groups=self.channels)
        lm2 = F.conv2d(img2, self.kernel_mean, padding=11//2, groups=self.channels)
        Ls = F.l1_loss(torch.clamp(img1 - lm1, 0, 1), torch.clamp(img2 - lm2, 0, 1))
        return 0.8 * L1 + 0.1 * Lhf + 0.1 * Ls

class L1SpecLoss(torch.nn.Module):
    def __init__(self):
        super(L1SpecLoss, self).__init__()
        self.kernel_mean = torch.ones((3, 1, 11, 11)) / (11**2)

    def forward(self, img1, img2):
        # L1 loss
        L1 = F.l1_loss(img1, img2)
        # spec loss
        if img1.is_cuda: self.kernel_mean = self.kernel_mean.cuda()
        lm1 = F.conv2d(img1, self.kernel_mean, padding=11//2, groups=3)
        lm2 = F.conv2d(img2, self.kernel_mean, padding=11//2, groups=3)
        Ls = F.l1_loss(torch.clamp(img1 - lm1, 0, 1), torch.clamp(img2 - lm2, 0, 1))
        return 0.8 * L1 + 0.2 * Ls

class L1HFENTemporalLoss(L1HFENLoss):
    # from https://research.nvidia.com/sites/default/files/publications/dnn_denoise_author.pdf
    def __init__(self):
        super(L1HFENTemporalLoss, self).__init__()

    def forward(self, img1, img2):
        assert img1.dim() > 4 and img2.dim() > 4, "Temporal loss needs 5D data!"
        # L1 loss
        L1 = F.l1_loss(img1[:, img1.size(1)//2], img2[:, img2.size(1)//2])
        # HFEN loss
        if img1.is_cuda: self.kernel = self.kernel.cuda()
        log1 = F.conv2d(img1[:, img1.size(1)//2], self.kernel, padding=2, groups=3)
        log2 = F.conv2d(img2[:, img2.size(1)//2], self.kernel, padding=2, groups=3)
        Lhf = F.l1_loss(log1, log2)
        # temporal loss
        Lt = 0
        for i in range(1, img1.size(1)):
            Lt += F.l1_loss(torch.abs(img1[:, i] - img1[:, i-1]), torch.abs(img2[:, i] - img2[:, i-1]))
        Lt /= max(1, img1.size(1) - 1)
        return 0.8 * L1 + 0.1 * Lhf + 0.1 * Lt