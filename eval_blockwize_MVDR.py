import os
import torch
import numpy as np
import shutil

device = torch.device(f'cuda:{n}' if torch.cuda.is_available() else 'cpu')
from stft_istft_real_imag import STFT
STFT = STFT(320, 160).to(device)


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

def lstm_conv_block(corr_mix):

    corr_accumulator = torch.zeros_like(corr_mix)
    for i in range(0, 31):
        rolled_corr = torch.roll(corr_mix, i, 1)
        corr_accumulator += rolled_corr
    corr_mean = corr_accumulator / 30
    return corr_mean

def complex_multi(Phi_N_inv_Phi_S, u1):

    real_part = torch.matmul(Phi_N_inv_Phi_S.real, u1.real) - torch.matmul(Phi_N_inv_Phi_S.imag, u1.imag)
    imag_part = torch.matmul(Phi_N_inv_Phi_S.real, u1.imag) + torch.matmul(Phi_N_inv_Phi_S.imag, u1.real)

    # Constructing the complex tensor from the real and imaginary parts
    result = torch.complex(real_part, imag_part)

    return result

def blockwize_mvdr_lstm(mix_spec, est_speech ,est_noise, NUM_MIC=5):
    # B, C, T, Frq, _ = mix_spec.shape
    spec_mix_t = mix_spec.permute([0, 3, 2, 1, 4])  # B,Frq,T,C,2
    corr_speech = compute_complex_covariance4(est_speech.permute(0,3,2,1,4)) # B,Frq,T,C,C
    corr_noise = compute_complex_covariance4(est_noise.permute(0,3,2,1,4)) # B,Frq,T,C,C
    corr_speech = lstm_conv_block(corr_speech)
    corr_noise = lstm_conv_block(corr_noise)
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
    est=est.permute(0,2,1,3)
    return est
