import torch as th
import torch.nn as nn

import torch.nn.functional as F
from nnstruct.stft import STFT
import torch
import pdb
def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class ConvTasNet(nn.Module):
    def __init__(self,
                 L=20,
                 N=513,
                 X=8,
                 R=4,
                 B=256,
                 H=512,
                 P=3,
                 norm="gLN",
                 num_spks=1,
                 non_linear="relu",
                 causal=False):
        super(ConvTasNet, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        self.ln = ChannelWiseLayerNorm(N)
        self.proj = Conv1D(N, B, 1)
        self.repeats = self._build_repeats(
            R,
            X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        self.mask = Conv1D(B, num_spks * N, 1)
        self.num_spks = num_spks
        self.STFT = STFT(1024, 256)


    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x):
        STFT_C1, phase_C1 = self.STFT.stft_mg(x)
        w = F.relu(STFT_C1.permute(0,2,1))
        y = self.proj(self.ln(w))
        y = self.repeats(y)
        e = th.chunk(self.mask(y), self.num_spks, 1)
        if self.non_linear_type == "softmax":
            m = self.non_linear(th.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(th.stack(e, dim=0))
        m2=m
        w2=w
        magn = [w2 * m2[n] for n in range(self.num_spks)]
        real = magn[0] * torch.cos(phase_C1.permute(0, 2, 1))
        imag = magn[0] * torch.sin(phase_C1.permute(0, 2, 1))
        est = torch.stack([real, imag], dim=-1)
        est=est.permute(0,3,2,1)
        y=self.STFT.istft(est)
        return y,m2

def foo_conv1d_block():
    nnet = Conv1DBlock(256, 512, 3, 20)
    print(param(nnet))


def foo_layernorm():
    C, T = 256, 20
    nnet1 = nn.LayerNorm([C, T], elementwise_affine=True)
    print(param(nnet1, Mb=False))
    nnet2 = nn.LayerNorm([C, T], elementwise_affine=False)
    print(param(nnet2, Mb=False))


def foo_conv_tas_net():
    x = th.rand(1, 16000)
    model = ConvTasNet(norm="gLN", causal=True)
    print("ConvTasNet #param: {:.2f}".format(param(model)))
    x = model(x)
    s1 = x[0]
    print(s1.shape)
    # from ptflops import get_model_complexity_info
    # with torch.no_grad():
    #     # 获取模型的计算复杂度信息
    #     macs, params = get_model_complexity_info(
    #         model, (16000,), as_strings=False, print_per_layer_stat=True, verbose=False
    #     )
    # macs = macs/1e9
    # params = params/1e6
    # # 打印 MACs 和参数量
    # print(f"MACs: {macs}")
    # print(f"Params: {params}")

if __name__ == "__main__":
    foo_conv_tas_net()
