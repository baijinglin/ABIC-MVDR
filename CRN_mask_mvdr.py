import torch.nn as nn
import torch.nn.functional as F
from stft_istft_real_imag import STFT
import torch
from config import *
import math
import pdb

import sys, os

# from util2_noise_5mic import mvdr_lstm
# sys.path.append(os.path.dirname(__file__))

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles



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



def complex_multi(Phi_N_inv_Phi_S, u1):

    real_part = torch.matmul(Phi_N_inv_Phi_S.real, u1.real) - torch.matmul(Phi_N_inv_Phi_S.imag, u1.imag)
    imag_part = torch.matmul(Phi_N_inv_Phi_S.real, u1.imag) + torch.matmul(Phi_N_inv_Phi_S.imag, u1.real)

    # Constructing the complex tensor from the real and imaginary parts
    result = torch.complex(real_part, imag_part)

    return result

def vector_unitization(x):
    amp = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + 1e-10)
    out = x / (amp + 1e-10)
    return out


class NET_Wrapper(nn.Module):
    def __init__(self,win_len,win_offset):
        super(NET_Wrapper, self).__init__()
        self.win_len = win_len
        self.win_offset = win_offset
        self.lstm_input_size = 256 * 4
        self.lstm_layers = 2
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(2, 3), stride=(1, 2))
        self.conv1_relu = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2))
        self.conv2_relu = nn.ELU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv3_relu = nn.ELU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2))
        self.conv4_relu = nn.ELU()
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2))
        self.conv5_relu = nn.ELU()
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_input_size,
                            num_layers=self.lstm_layers,
                            batch_first=True)

        self.conv5_t = nn.ConvTranspose2d(in_channels=256 * 2, out_channels=128, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv5_t_relu = nn.ELU()
        self.conv4_t = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv4_t_relu = nn.ELU()
        self.conv3_t = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv3_t_relu = nn.ELU()
        self.conv2_t = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 3), stride=(1, 2),
                                          output_padding=(0, 1), padding=(1, 0))
        self.conv2_t_relu = nn.ELU()
        self.conv1_t = nn.ConvTranspose2d(in_channels=16 * 2, out_channels=1, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv1_t_relu = nn.ELU()
        # nn.Softplus()
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5_bn = nn.BatchNorm2d(256)

        self.conv5_t_bn = nn.BatchNorm2d(128)
        self.conv4_t_bn = nn.BatchNorm2d(64)
        self.conv3_t_bn = nn.BatchNorm2d(32)
        self.conv2_t_bn = nn.BatchNorm2d(16)
        self.conv1_t_bn = nn.BatchNorm2d(1)
        ################################################
        self.conv5_t_key1 = nn.ConvTranspose2d(in_channels=256 * 2, out_channels=128, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv4_t_key1 = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv3_t_key1 = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv2_t_key1 = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 3), stride=(1, 2),
                                          output_padding=(0, 1), padding=(1, 0))
        self.conv1_t_key1 = nn.ConvTranspose2d(in_channels=16 * 2, out_channels=24, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv5_t_bn_key1 = nn.BatchNorm2d(128)
        self.conv4_t_bn_key1 = nn.BatchNorm2d(64)
        self.conv3_t_bn_key1 = nn.BatchNorm2d(32)
        self.conv2_t_bn_key1 = nn.BatchNorm2d(16)
        self.conv1_t_bn_key1 = nn.BatchNorm2d(24)

        self.conv5_t_query1 = nn.ConvTranspose2d(in_channels=256 * 2, out_channels=128, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv4_t_query1 = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv3_t_query1 = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv2_t_query1 = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 3), stride=(1, 2),
                                          output_padding=(0, 1), padding=(1, 0))
        self.conv1_t_query1 = nn.ConvTranspose2d(in_channels=16 * 2, out_channels=24, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv5_t_bn_query1 = nn.BatchNorm2d(128)
        self.conv4_t_bn_query1 = nn.BatchNorm2d(64)
        self.conv3_t_bn_query1 = nn.BatchNorm2d(32)
        self.conv2_t_bn_query1 = nn.BatchNorm2d(16)
        self.conv1_t_bn_query1 = nn.BatchNorm2d(24)

        self.conv5_t_key2 = nn.ConvTranspose2d(in_channels=256 * 2, out_channels=128, kernel_size=(2, 3), stride=(1, 2),
                                               padding=(1, 0))
        self.conv4_t_key2 = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                               padding=(1, 0))
        self.conv3_t_key2 = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 3), stride=(1, 2),
                                               padding=(1, 0))
        self.conv2_t_key2 = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 3), stride=(1, 2),
                                               output_padding=(0, 1), padding=(1, 0))
        self.conv1_t_key2 = nn.ConvTranspose2d(in_channels=16 * 2, out_channels=24, kernel_size=(2, 3), stride=(1, 2),
                                               padding=(1, 0))
        self.conv5_t_bn_key2 = nn.BatchNorm2d(128)
        self.conv4_t_bn_key2 = nn.BatchNorm2d(64)
        self.conv3_t_bn_key2 = nn.BatchNorm2d(32)
        self.conv2_t_bn_key2 = nn.BatchNorm2d(16)
        self.conv1_t_bn_key2 = nn.BatchNorm2d(24)

        self.conv5_t_query2 = nn.ConvTranspose2d(in_channels=256 * 2, out_channels=128, kernel_size=(2, 3),
                                                 stride=(1, 2),
                                                 padding=(1, 0))
        self.conv4_t_query2 = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=(2, 3),
                                                 stride=(1, 2),
                                                 padding=(1, 0))
        self.conv3_t_query2= nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 3), stride=(1, 2),
                                                 padding=(1, 0))
        self.conv2_t_query2 = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 3), stride=(1, 2),
                                                 output_padding=(0, 1), padding=(1, 0))
        self.conv1_t_query2 = nn.ConvTranspose2d(in_channels=16 * 2, out_channels=24, kernel_size=(2, 3), stride=(1, 2),
                                                 padding=(1, 0))
        self.conv5_t_bn_query2 = nn.BatchNorm2d(128)
        self.conv4_t_bn_query2 = nn.BatchNorm2d(64)
        self.conv3_t_bn_query2 = nn.BatchNorm2d(32)
        self.conv2_t_bn_query2 = nn.BatchNorm2d(16)
        self.conv1_t_bn_query2 = nn.BatchNorm2d(24)
        ################################################
        self.pad = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.STFT = STFT(self.win_len, self.win_offset)
        self.fc1 = nn.Linear(in_features=161, out_features=161)
        self.fc2 = nn.Linear(in_features=161, out_features=161)
        self.fc3 = nn.Linear(in_features=161, out_features=161)
        self.fc4 = nn.Linear(in_features=161, out_features=161)
        self.fc5 = nn.Linear(in_features=161, out_features=161)
        self.sigmod = nn.Sigmoid()
        self.ac = nn.Tanh()
    def forward(self, inputs):
        [batch_size, _, channel] = inputs.shape
        mix_data = inputs.permute([0, 2, 1]).reshape(batch_size * channel, -1)
        mix_spec = self.STFT.transform(mix_data)
        mix_spec = mix_spec.reshape(batch_size, channel, -1, FFT_SIZE, 2)
        input_data = torch.cat([mix_spec[..., 0], mix_spec[..., 1]], 1)
        e1 = self.conv1_relu(self.conv1_bn(self.conv1(self.pad(input_data))))
        e2 = self.conv2_relu(self.conv2_bn(self.conv2(self.pad(e1))))
        e3 = self.conv3_relu(self.conv3_bn(self.conv3(self.pad(e2))))
        e4 = self.conv4_relu(self.conv4_bn(self.conv4(self.pad(e3))))
        e5 = self.conv5_relu(self.conv5_bn(self.conv5(self.pad(e4))))

        self.lstm.flatten_parameters()
        out_real = e5.contiguous().transpose(1, 2)
        out_real = out_real.contiguous().view(out_real.size(0), out_real.size(1), -1)
        lstm_out, _ = self.lstm(out_real)
        lstm_out_real = lstm_out.contiguous().view(lstm_out.size(0), lstm_out.size(1), 256, 4)
        lstm_out_real = lstm_out_real.contiguous().transpose(1, 2)

        t5 = self.conv5_t_relu(self.conv5_t_bn(self.conv5_t(self.pad(torch.cat((lstm_out_real, e5), dim=1)))))
        t4 = self.conv4_t_relu(self.conv4_t_bn(self.conv4_t(self.pad(torch.cat((t5, e4), dim=1)))))
        t3 = self.conv3_t_relu(self.conv3_t_bn(self.conv3_t(self.pad(torch.cat((t4, e3), dim=1)))))
        t2 = self.conv2_t_relu(self.conv2_t_bn(self.conv2_t(self.pad(torch.cat((t3, e2), dim=1)))))
        t1 = self.conv1_t_relu(self.conv1_t_bn(self.conv1_t(self.pad(torch.cat((t2, e1), dim=1)))))
        speech_mask = self.sigmod(self.fc5(t1))
        ########################
        t5_q1 = self.conv5_t_relu(self.conv5_t_bn_query1(self.conv5_t_query1(self.pad(torch.cat((lstm_out_real, e5), dim=1)))))
        t4_q1 = self.conv4_t_relu(self.conv4_t_bn_query1(self.conv4_t_query1(self.pad(torch.cat((t5_q1, e4), dim=1)))))
        t3_q1 = self.conv3_t_relu(self.conv3_t_bn_query1(self.conv3_t_query1(self.pad(torch.cat((t4_q1, e3), dim=1)))))
        t2_q1 = self.conv2_t_relu(self.conv2_t_bn_query1(self.conv2_t_query1(self.pad(torch.cat((t3_q1, e2), dim=1)))))
        t1_q1 = self.conv1_t_relu(self.conv1_t_bn_query1(self.conv1_t_query1(self.pad(torch.cat((t2_q1, e1), dim=1)))))
        query_1 = self.ac(self.fc1(t1_q1))

        t5_k1 = self.conv5_t_relu(self.conv5_t_bn_key1(self.conv5_t_key1(self.pad(torch.cat((lstm_out_real, e5), dim=1)))))
        t4_k1 = self.conv4_t_relu(self.conv4_t_bn_key1(self.conv4_t_key1(self.pad(torch.cat((t5_k1, e4), dim=1)))))
        t3_k1 = self.conv3_t_relu(self.conv3_t_bn_key1(self.conv3_t_key1(self.pad(torch.cat((t4_k1, e3), dim=1)))))
        t2_k1 = self.conv2_t_relu(self.conv2_t_bn_key1(self.conv2_t_key1(self.pad(torch.cat((t3_k1, e2), dim=1)))))
        t1_k1 = self.conv1_t_relu(self.conv1_t_bn_key1(self.conv1_t_key1(self.pad(torch.cat((t2_k1, e1), dim=1)))))
        key_1 = self.ac(self.fc2(t1_k1))

        t5_q2 = self.conv5_t_relu(self.conv5_t_bn_query2(self.conv5_t_query2(self.pad(torch.cat((lstm_out_real, e5), dim=1)))))
        t4_q2 = self.conv4_t_relu(self.conv4_t_bn_query2(self.conv4_t_query2(self.pad(torch.cat((t5_q2, e4), dim=1)))))
        t3_q2 = self.conv3_t_relu(self.conv3_t_bn_query2(self.conv3_t_query2(self.pad(torch.cat((t4_q2, e3), dim=1)))))
        t2_q2 = self.conv2_t_relu(self.conv2_t_bn_query2(self.conv2_t_query2(self.pad(torch.cat((t3_q2, e2), dim=1)))))
        t1_q2 = self.conv1_t_relu(self.conv1_t_bn_query2(self.conv1_t_query2(self.pad(torch.cat((t2_q2, e1), dim=1)))))
        query_2 = self.ac(self.fc3(t1_q2))

        t5_k2 = self.conv5_t_relu(self.conv5_t_bn_key2(self.conv5_t_key2(self.pad(torch.cat((lstm_out_real, e5), dim=1)))))
        t4_k2 = self.conv4_t_relu(self.conv4_t_bn_key2(self.conv4_t_key2(self.pad(torch.cat((t5_k2, e4), dim=1)))))
        t3_k2 = self.conv3_t_relu(self.conv3_t_bn_key2(self.conv3_t_key2(self.pad(torch.cat((t4_k2, e3), dim=1)))))
        t2_k2 = self.conv2_t_relu(self.conv2_t_bn_key2(self.conv2_t_key2(self.pad(torch.cat((t3_k2, e2), dim=1)))))
        t1_k2 = self.conv1_t_relu(self.conv1_t_bn_key2(self.conv1_t_key2(self.pad(torch.cat((t2_k2, e1), dim=1)))))
        key_2 = self.ac(self.fc4(t1_k2))
        spec_enhance = mvdr_lstm(mix_spec, speech_mask, [query_1, key_1, query_2, key_2], 5)
        est_speech = self.STFT.inverse(spec_enhance)
        return  est_speech


def mvdr_lstm( spec_mix, speech_mask, est_embedding,  NUM_MIC=5):
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

    value_speech= corr_speech.reshape(B * Frq, T,-1)
    value_noise = corr_noise.reshape(B * Frq, T, -1)
    #####corr_speech mask
    query1 = est_embedding[0].permute([0, 3, 2, 1]).reshape(B * Frq, T, -1)
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
    ######corr_noise mask
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
    W_t_f = (a1) / (trace_Phi_N_inv_Phi_S.unsqueeze(-1).unsqueeze(-1) + epsilon)
    W_t_f = W_t_f.permute(0, 1, 2, 4, 3)
    W_t_f_H = torch.conj(W_t_f)
    stft_mix = torch.view_as_complex(torch.stack([spec_mix_t[..., 0], spec_mix_t[..., 1]], -1)).unsqueeze(-1)
    S_hat_t_f = complex_multi(W_t_f_H, stft_mix).squeeze(-1).squeeze(-1)
    est=torch.stack([S_hat_t_f.real, S_hat_t_f.imag],dim=-1)
    ##################
    est=est.permute(0,2,1,3)
    return est

def main():
    # 设定输入变量的维度
    B = 1  # 批次大小，可以根据需要调整
    T = 100  # 时间维度的大小
    F = 161  # 频率维度的大小

    # 随机生成一个输入变量
    inputs = torch.randn(B,16000,5).cuda()

    # 创建ICRN模型实例
    CRN = NET_Wrapper(win_len=320, win_offset=160).cuda()

    # 将输入传递给模型并获取输出
    outputs = CRN(inputs)

    # 打印输出以进行调试
    print("输出形状:", outputs.shape)

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles

def complexity():
    model = NET_Wrapper(win_len=320, win_offset=160)
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
    complexity()

