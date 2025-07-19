import torch.nn as nn
import torch.nn.functional as F
from stft_istft_real_imag import STFT
import torch
import math
import sys, os
import pdb

# from util2_noise_5mic import mvdr_lstm
# sys.path.append(os.path.dirname(__file__))

class convGLU(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=(5, 1), stride=(1, 1),
                 padding=(2, 0), dilation=1, groups=1, bias=True):
        super(convGLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.convGate = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        #         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.gate_act = nn.Sigmoid()

    def forward(self, inputs):
        outputs = self.conv(inputs)  # *self.gate_act(self.convGate(inputs))
        return outputs


class convTransGLU(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=(5, 1), stride=(1, 1),
                 padding=(2, 0), output_padding=(0, 0), dilation=1, groups=1, bias=True):
        super(convTransGLU, self).__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, output_padding=output_padding,
                                            dilation=dilation, groups=groups, bias=bias)
        # self.convTransGate = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        #         stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        # self.gate_act = nn.Sigmoid()

    def forward(self, inputs):
        outputs = self.convTrans(inputs)  # *self.gate_act(self.convTransGate(inputs))
        return outputs

def complex_multi(Phi_N_inv_Phi_S, u1):

    real_part = torch.matmul(Phi_N_inv_Phi_S.real, u1.real) - torch.matmul(Phi_N_inv_Phi_S.imag, u1.imag)
    imag_part = torch.matmul(Phi_N_inv_Phi_S.real, u1.imag) + torch.matmul(Phi_N_inv_Phi_S.imag, u1.real)

    # Constructing the complex tensor from the real and imaginary parts
    result = torch.complex(real_part, imag_part)

    return result

class ICRN(nn.Module):
    def __init__(self, in_ch=10, out_ch=2, channels=24, lstm_hiden=48):  # 32-64
        super(ICRN, self).__init__()
        self.act = nn.ELU()
        self.sigmod = nn.Sigmoid()
        self.e1 = convGLU(in_ch, channels)
        self.e2 = convGLU(channels, channels)
        self.e3 = convGLU(channels, channels)
        self.e4 = convGLU(channels, channels)
        self.e5 = convGLU(channels, channels)
        self.e6 = convGLU(channels, channels)
        self.ac = nn.Tanh()

        self.BNe1 = nn.BatchNorm2d(channels)
        self.BNe2 = nn.BatchNorm2d(channels)
        self.BNe3 = nn.BatchNorm2d(channels)
        self.BNe4 = nn.BatchNorm2d(channels)
        self.BNe5 = nn.BatchNorm2d(channels)
        self.BNe6 = nn.BatchNorm2d(channels)

        self.d_mask6 = convTransGLU(2 * channels, channels)
        self.d_mask5 = convTransGLU(2 * channels, channels)
        self.d_mask4= convTransGLU(2 * channels, channels)
        self.d_mask3 = convTransGLU(2 * channels, channels)
        self.d_mask2= convTransGLU(2 * channels, channels)
        self.d_mask1 = convTransGLU(2 * channels, 5)

        self.BNd_mask6 = nn.BatchNorm2d(channels)
        self.BNd_mask5 = nn.BatchNorm2d(channels)
        self.BNd_mask4 = nn.BatchNorm2d(channels)
        self.BNd_mask3 = nn.BatchNorm2d(channels)
        self.BNd_mask2 = nn.BatchNorm2d(channels)
        self.BNd_mask1 = nn.BatchNorm2d(5)
        ###################
        self.lstm = nn.LSTM(channels, lstm_hiden,
                            num_layers=2, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(in_features=161, out_features=161)
        self.linear_lstm_out = nn.Linear(lstm_hiden, channels)
        self.STFT = STFT(320, 160)

    def forward(self, inputs):
        [batch_size, _, channel] = inputs.shape
        mix_data = inputs.permute([0, 2, 1]).reshape(batch_size * channel, -1)
        mix_spec = self.STFT.transform(mix_data)
        mix_spec = mix_spec.reshape(batch_size, channel, -1, 161, 2)
        input_data = torch.cat([mix_spec[..., 0], mix_spec[..., 1]], 1)
        input_data = input_data.permute(0, 1, 3, 2)
        e1 = self.act(self.BNe1(self.e1(input_data)))
        e2 = self.act(self.BNe2(self.e2(e1)))
        e3 = self.act(self.BNe3(self.e3(e2)))
        e4 = self.act(self.BNe4(self.e4(e3)))
        e5 = self.act(self.BNe5(self.e5(e4)))
        e6 = self.act(self.BNe6(self.e6(e5)))

        shape_in = e6.shape
        lstm_in = e6.permute(0, 2, 3, 1).reshape(-1, shape_in[3], shape_in[1])
        lstm_out, _ = self.lstm(lstm_in.float())
        lstm_out = self.linear_lstm_out(lstm_out)
        lstm_out = lstm_out.reshape(shape_in[0], shape_in[2], shape_in[3], shape_in[1]).permute(0, 3, 1, 2)
        d6_mask = self.act(self.BNd_mask6(self.d_mask6(torch.cat([e6, lstm_out], dim=1))))
        d5_mask = self.act(self.BNd_mask5(self.d_mask5(torch.cat([e5, d6_mask], dim=1))))
        d4_mask = self.act(self.BNd_mask4(self.d_mask4(torch.cat([e4, d5_mask], dim=1))))
        d3_mask = self.act(self.BNd_mask3(self.d_mask3(torch.cat([e3, d4_mask], dim=1))))
        d2_mask = self.act(self.BNd_mask2(self.d_mask2(torch.cat([e2, d3_mask], dim=1))))
        d1_mask = self.act(self.BNd_mask1(self.d_mask1(torch.cat([e1, d2_mask], dim=1))))

        speech_mask =  self.sigmod(self.fc(d1_mask.permute(0,1,3,2)))  # (B, feature, T, F)

        ########################
        mix_real = mix_spec[..., 0]
        mix_imag = mix_spec[..., 1]
        mix_magnitude = torch.sqrt(mix_real ** 2 + mix_imag ** 2)
        mix_phase = torch.atan2(mix_imag, mix_real)
        mix_magnitude_mask = mix_magnitude * speech_mask
        est_real = mix_magnitude_mask * torch.cos(mix_phase)
        est_imag = mix_magnitude_mask * torch.sin(mix_phase)
        est_speech = torch.stack([est_real, est_imag], -1)  # B,Frq,T,C,2

        B,channel,T,F,_ = est_speech.shape
        est_speech2 = est_speech.reshape(B*channel,T,F,-1)
        est_speech3 = self.STFT.inverse(est_speech2).reshape(B,channel,-1).permute(0,2,1)
        return est_speech3, est_speech, mix_spec


class loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_r, x_i = x[:, 0], x[:, 1]
        y_r, y_i = y[:, 0], y[:, 1]
        x_a = torch.sqrt(x_r ** 2 + x_i ** 2 + 1e-10)
        y_a = torch.sqrt(y_r ** 2 + y_i ** 2 + 1e-10)
        loss = torch.abs(x_r - y_r).mean() + torch.abs(x_i - y_i).mean() + torch.abs(x_a - y_a).mean()
        return loss


def vector_unitization(x):
    amp = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + 1e-10)
    out = x / (amp + 1e-10)
    return out


def main():
    # 设定输入变量的维度
    B = 1  # 批次大小，可以根据需要调整
    T = 100  # 时间维度的大小
    F = 161  # 频率维度的大小

    # 随机生成一个输入变量
    inputs = torch.randn(B,16000,5)

    # 创建ICRN模型实例
    model = ICRN(in_ch=10)  # 确保输入通道数与模型匹配

    # 将输入传递给模型并获取输出
    outputs = model(inputs)
    #打印输出以进行调试
    print("输出形状:", outputs.shape)
    from ptflops import get_model_complexity_info
    with torch.no_grad():
        # 获取模型的计算复杂度信息
        macs, params = get_model_complexity_info(
            model, (16000, 5), as_strings=False, print_per_layer_stat=True, verbose=False
        )
    macs = macs / 1e9
    params = params / 1e6
    # 打印 MACs 和参数量
    print(f"MACs: {macs}")
    print(f"Params: {params}")


if __name__ == '__main__':
    main()
# def complexity():
#     from thop import profile, clever_format
#     inputs = torch.randn(1,4,100,161).cuda()
#     model = ICRN(4).cuda()
#     total_ops, total_params = profile(model, inputs=(inputs,), verbose=False)
#     flops, params = clever_format([total_ops, total_params], "%.3f ")
#     print(flops, params)
#     # mac, param = thop.profile(model, inputs=(inputs,))
#     # print('mac:',mac/(2**30),' param', param/(2**20))

# if __name__ == '__main__':
#     complexity()
