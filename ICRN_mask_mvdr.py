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
        # pdb.set_trace()
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
    
def compute_complex_covariance4(stft_result):
    stft_complex = torch.view_as_complex(stft_result)
    mean = stft_complex.mean(dim=-3, keepdim=True)
    stft_complex_demeaned = stft_complex - mean
    stft_complex_demeaned = stft_complex_demeaned.unsqueeze(-1)  # Shape: (B, F, T, M, 1)

    # Compute the covariance matrix using batch operations
    stft_complex_demeaned_conj = stft_complex_demeaned.conj().transpose(-2, -1)  # Shape: (B, F, T, 1, M)
    complex_cov_per_time = torch.matmul(stft_complex_demeaned, stft_complex_demeaned_conj)  # Shape: (B, F, T, M, M)

    # Normalize the covariance matrix
    normalization_factor = stft_complex_demeaned.shape[1] - 1
    complex_cov_per_time = complex_cov_per_time / normalization_factor

    return complex_cov_per_time

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

        self.d16 = convTransGLU(2 * channels, channels)
        self.d15 = convTransGLU(2 * channels, channels)
        self.d14 = convTransGLU(2 * channels, channels)
        self.d13 = convTransGLU(2 * channels, channels)
        self.d12 = convTransGLU(2 * channels, channels)
        self.d11 = convTransGLU(2 * channels, 24)

        self.d26 = convTransGLU(2 * channels, channels)
        self.d25 = convTransGLU(2 * channels, channels)
        self.d24 = convTransGLU(2 * channels, channels)
        self.d23 = convTransGLU(2 * channels, channels)
        self.d22 = convTransGLU(2 * channels, channels)
        self.d21 = convTransGLU(2 * channels, 24)

        self.BNe1 = nn.BatchNorm2d(channels)
        self.BNe2 = nn.BatchNorm2d(channels)
        self.BNe3 = nn.BatchNorm2d(channels)
        self.BNe4 = nn.BatchNorm2d(channels)
        self.BNe5 = nn.BatchNorm2d(channels)
        self.BNe6 = nn.BatchNorm2d(channels)

        self.BNd16 = nn.BatchNorm2d(channels)
        self.BNd15 = nn.BatchNorm2d(channels)
        self.BNd14 = nn.BatchNorm2d(channels)
        self.BNd13 = nn.BatchNorm2d(channels)
        self.BNd12 = nn.BatchNorm2d(channels)
        self.BNd11 = nn.BatchNorm2d(24)
        
        self.BNd26 = nn.BatchNorm2d(channels)
        self.BNd25 = nn.BatchNorm2d(channels)
        self.BNd24 = nn.BatchNorm2d(channels)
        self.BNd23 = nn.BatchNorm2d(channels)
        self.BNd22 = nn.BatchNorm2d(channels)
        self.BNd21 = nn.BatchNorm2d(24)

        self.ac = nn.Tanh()

        self.fc1 = nn.Linear(in_features=161, out_features=161)
        self.fc2 = nn.Linear(in_features=161, out_features=161)
        #####################
        self.d36 = convTransGLU(2 * channels, channels)
        self.d35 = convTransGLU(2 * channels, channels)
        self.d34 = convTransGLU(2 * channels, channels)
        self.d33 = convTransGLU(2 * channels, channels)
        self.d32 = convTransGLU(2 * channels, channels)
        self.d31 = convTransGLU(2 * channels, 24)

        self.d46 = convTransGLU(2 * channels, channels)
        self.d45 = convTransGLU(2 * channels, channels)
        self.d44 = convTransGLU(2 * channels, channels)
        self.d43 = convTransGLU(2 * channels, channels)
        self.d42 = convTransGLU(2 * channels, channels)
        self.d41 = convTransGLU(2 * channels, 24)

        self.BNd16_2 = nn.BatchNorm2d(channels)
        self.BNd15_2 = nn.BatchNorm2d(channels)
        self.BNd14_2 = nn.BatchNorm2d(channels)
        self.BNd13_2 = nn.BatchNorm2d(channels)
        self.BNd12_2 = nn.BatchNorm2d(channels)
        self.BNd11_2 = nn.BatchNorm2d(24)

        self.BNd26_2 = nn.BatchNorm2d(channels)
        self.BNd25_2 = nn.BatchNorm2d(channels)
        self.BNd24_2 = nn.BatchNorm2d(channels)
        self.BNd23_2 = nn.BatchNorm2d(channels)
        self.BNd22_2 = nn.BatchNorm2d(channels)
        self.BNd21_2 = nn.BatchNorm2d(24)
        self.fc3 = nn.Linear(in_features=161, out_features=161)
        self.fc4 = nn.Linear(in_features=161, out_features=161)
        ####################
        self.d_mask6 = convTransGLU(2 * channels, channels)
        self.d_mask5 = convTransGLU(2 * channels, channels)
        self.d_mask4= convTransGLU(2 * channels, channels)
        self.d_mask3 = convTransGLU(2 * channels, channels)
        self.d_mask2= convTransGLU(2 * channels, channels)
        self.d_mask1 = convTransGLU(2 * channels, 1)

        self.BNd_mask6 = nn.BatchNorm2d(channels)
        self.BNd_mask5 = nn.BatchNorm2d(channels)
        self.BNd_mask4 = nn.BatchNorm2d(channels)
        self.BNd_mask3 = nn.BatchNorm2d(channels)
        self.BNd_mask2 = nn.BatchNorm2d(channels)
        self.BNd_mask1 = nn.BatchNorm2d(1)
        self.ac = nn.Tanh()
        self.fc5 = nn.Linear(in_features=161, out_features=161)
        ###################
        self.lstm = nn.LSTM(channels, lstm_hiden,
                            num_layers=2, batch_first=True, bidirectional=False)

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

        d16 = self.act(self.BNd16(self.d16(torch.cat([e6, lstm_out], dim=1))))
        d15 = self.act(self.BNd15(self.d15(torch.cat([e5, d16], dim=1))))
        d14 = self.act(self.BNd14(self.d14(torch.cat([e4, d15], dim=1))))
        d13 = self.act(self.BNd13(self.d13(torch.cat([e3, d14], dim=1))))
        d12 = self.act(self.BNd12(self.d12(torch.cat([e2, d13], dim=1))))
        d11 = self.act(self.BNd11(self.d11(torch.cat([e1, d12], dim=1))))
        query = self.ac(self.fc1(d11.permute(0,1,3,2)))

        d26 = self.act(self.BNd26(self.d26(torch.cat([e6, lstm_out], dim=1))))
        d25 = self.act(self.BNd25(self.d25(torch.cat([e5, d26], dim=1))))
        d24 = self.act(self.BNd24(self.d24(torch.cat([e4, d25], dim=1))))
        d23 = self.act(self.BNd23(self.d23(torch.cat([e3, d24], dim=1))))
        d22 = self.act(self.BNd22(self.d22(torch.cat([e2, d23], dim=1))))
        d21 = self.act(self.BNd21(self.d21(torch.cat([e1, d22], dim=1))))
        key = self.ac(self.fc2(d21.permute(0,1,3,2)))  # (B, feature, T, F)

        ########################
        d16_2 = self.act(self.BNd16_2(self.d36(torch.cat([e6, lstm_out], dim=1))))
        d15_2 = self.act(self.BNd15_2(self.d35(torch.cat([e5, d16_2], dim=1))))
        d14_2 = self.act(self.BNd14_2(self.d34(torch.cat([e4, d15_2], dim=1))))
        d13_2 = self.act(self.BNd13_2(self.d33(torch.cat([e3, d14_2], dim=1))))
        d12_2 = self.act(self.BNd12_2(self.d32(torch.cat([e2, d13_2], dim=1))))
        d11_2 = self.act(self.BNd11_2(self.d31(torch.cat([e1, d12_2], dim=1))))
        query_2 = self.ac(self.fc3(d11_2.permute(0,1,3,2)))

        d26_2 = self.act(self.BNd26_2(self.d46(torch.cat([e6, lstm_out], dim=1))))
        d25_2 = self.act(self.BNd25_2(self.d45(torch.cat([e5, d26_2], dim=1))))
        d24_2 = self.act(self.BNd24_2(self.d44(torch.cat([e4, d25_2], dim=1))))
        d23_2 = self.act(self.BNd23_2(self.d43(torch.cat([e3, d24_2], dim=1))))
        d22_2 = self.act(self.BNd22_2(self.d42(torch.cat([e2, d23_2], dim=1))))
        d21_2 = self.act(self.BNd21_2(self.d41(torch.cat([e1, d22_2], dim=1))))
        key_2 = self.ac(self.fc4(d21_2.permute(0,1,3,2)))  # (B, feature, T, F)
        
        d6_mask = self.act(self.BNd_mask6(self.d_mask6(torch.cat([e6, lstm_out], dim=1))))
        d5_mask = self.act(self.BNd_mask5(self.d_mask5(torch.cat([e5, d6_mask], dim=1))))
        d4_mask = self.act(self.BNd_mask4(self.d_mask4(torch.cat([e4, d5_mask], dim=1))))
        d3_mask = self.act(self.BNd_mask3(self.d_mask3(torch.cat([e3, d4_mask], dim=1))))
        d2_mask = self.act(self.BNd_mask2(self.d_mask2(torch.cat([e2, d3_mask], dim=1))))
        d1_mask = self.act(self.BNd_mask1(self.d_mask1(torch.cat([e1, d2_mask], dim=1))))
        speech_mask =  self.sigmod(self.fc5(d1_mask.permute(0,1,3,2)))  # (B, feature, T, F)
        
        spec_enhance, est_speech_mask = mvdr_lstm(mix_spec ,speech_mask,[query, key, query_2, key_2], 5)
        est_speech = self.STFT.inverse(spec_enhance)
        return est_speech


def mvdr_lstm( spec_mix, speech_mask, est_embedding,  NUM_MIC=5):
    device = spec_mix.device
    speech_mask = speech_mask.permute([0, 3, 2, 1])
    B, C, T, Frq, _ = spec_mix.shape
    spec_mix_t = spec_mix.permute([0, 3, 2, 1, 4]) # B,Frq,T,C,2
    mix_real = spec_mix_t[..., 0]
    mix_imag = spec_mix_t[..., 1]
    mix_magnitude = torch.sqrt(mix_real ** 2 + mix_imag ** 2)
    mix_phase = torch.atan2(mix_imag, mix_real)
    mix_magnitude_mask = mix_magnitude * speech_mask
    est_real =  mix_magnitude_mask * torch.cos(mix_phase)
    est_imag =  mix_magnitude_mask * torch.sin(mix_phase)
    est_speech = torch.stack([est_real, est_imag], -1)# B,Frq,T,C,2
    est_noise = spec_mix_t - est_speech # B,Frq,T,C,2
    corr_speech = compute_complex_covariance4(est_speech) # B,Frq,T,C,C
    corr_noise = compute_complex_covariance4(est_noise) # B,Frq,T,C,C
    value_speech= corr_speech.reshape(B*Frq, T,-1)
    value_noise = corr_noise.reshape(B * Frq, T, -1)
    
    #####corr_speech 
    query1 = est_embedding[0].permute([0, 3, 2, 1]).reshape(B*Frq, T, -1)
    key1 = est_embedding[1].permute([0, 3, 2, 1]).reshape(B * Frq, T, -1)
    d_k = query1.size(-1)
    scores1 = torch.matmul(query1, key1.transpose(-2, -1)) / math.sqrt(d_k)
    mask1 = (1 - torch.tril(torch.ones_like(scores1))).type(torch.bool)
    scores1 = scores1.masked_fill(mask1, float('-inf'))
    p_attn1 = torch.softmax(scores1, dim=-1)
    corr_real1 = torch.matmul(p_attn1, value_speech.real)
    corr_imag1 = torch.matmul(p_attn1, value_speech.imag)
    corr1 = torch.view_as_complex(torch.stack([corr_real1, corr_imag1], -1))
    corr_speech = corr1.reshape(B * Frq, T, NUM_MIC, NUM_MIC)
    corr_speech = corr_speech.reshape(B , Frq, T, NUM_MIC, NUM_MIC)
    
    ######corr_noise
    query2 = est_embedding[2].permute([0, 3, 2, 1]).reshape(B*Frq, T, -1)
    key2 = est_embedding[3].permute([0, 3, 2, 1]).reshape(B * Frq, T, -1)
    d_k2 = query2.size(-1)
    scores2 = torch.matmul(query2, key2.transpose(-2, -1)) / math.sqrt(d_k2)
    mask2 = (1 - torch.tril(torch.ones_like(scores2))).type(torch.bool)
    scores2 = scores2.masked_fill(mask2, float('-inf'))
    p_attn2 = torch.softmax(scores2, dim=-1)
    corr_real2 = torch.matmul(p_attn2,  value_noise.real)
    corr_imag2 = torch.matmul(p_attn2,  value_noise.imag)
    corr2 = torch.view_as_complex(torch.stack([corr_real2, corr_imag2], -1))
    corr_noise = corr2.reshape(B * Frq, T, NUM_MIC, NUM_MIC)
    corr_noise = corr_noise.reshape(B ,Frq, T, NUM_MIC, NUM_MIC)
    device = corr_noise.device
    u1 = torch.tensor([1, 0,0,0,0], dtype=torch.cfloat).to(device)
    Phi_N= corr_noise
    Phi_S= corr_speech
    B, F, T, _, _ = Phi_S.shape
    # import pdb
    # pdb.set_trace()
    # import scipy.io as sio
    # sio.savemat('data2/Phi_S_fixed.mat', {'corr': Phi_S.data.cpu().numpy()})
    # sio.savemat('data2/Phi_N_fixed.mat', {'corr': Phi_N.data.cpu().numpy()})
    # Phi_N_inv = complex_number_inverse(Phi_N)
    epsilon=1e-7
    eye_complex = torch.eye(5, device=Phi_N.device) + 1j * torch.eye(5, device=Phi_N.device)
    eye_complex = eye_complex.expand_as(Phi_N)
    Phi_N = Phi_N + epsilon * eye_complex
    Phi_N_inv = torch.inverse(Phi_N)
    Phi_N_inv_Phi_S = complex_multi(Phi_N_inv, Phi_S)
    trace_Phi_N_inv_Phi_S = torch.einsum('...ii->...', Phi_N_inv_Phi_S)

    u1 = u1.expand(B, F, T, 5).unsqueeze(-1)
    a1 = complex_multi(Phi_N_inv_Phi_S, u1) #molecule
    # 获取数据类型的最小正数 epsilon
    epsilon_value = torch.finfo(trace_Phi_N_inv_Phi_S.dtype).eps
    # 创建一个复数，其实部和虚部都是 epsilon_value
    epsilon = torch.tensor(epsilon_value) + 1j * torch.tensor(epsilon_value)
    # import pdb
    # pdb.set_trace()
    W_t_f = (a1) / (trace_Phi_N_inv_Phi_S.unsqueeze(-1).unsqueeze(-1) + epsilon)
    W_t_f = W_t_f.permute(0, 1, 2, 4, 3)
    W_t_f_H = torch.conj(W_t_f)
    stft_mix = torch.view_as_complex(torch.stack([spec_mix_t[..., 0], spec_mix_t[..., 1]], -1)).unsqueeze(-1)
    S_hat_t_f = complex_multi(W_t_f_H, stft_mix).squeeze(-1).squeeze(-1)
    est=torch.stack([S_hat_t_f.real, S_hat_t_f.imag],dim=-1)
    ##################
    est=est.permute(0,2,1,3)
    return est, est_speech


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
