import torch.autograd.variable
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.cuda as cuda
import math
import shutil
import pdb
import torch.nn.init as init
from nnstruct.stft import STFT
from nnstruct.transformer_encoder import TransformerEncoder

def complex_multi(Phi_N_inv_Phi_S, u1):

    real_part = torch.matmul(Phi_N_inv_Phi_S.real, u1.real) - torch.matmul(Phi_N_inv_Phi_S.imag, u1.imag)
    imag_part = torch.matmul(Phi_N_inv_Phi_S.real, u1.imag) + torch.matmul(Phi_N_inv_Phi_S.imag, u1.real)

    # Constructing the complex tensor from the real and imaginary parts
    result = torch.complex(real_part, imag_part)

    return result

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

class Net_atten(nn.Module):
    def __init__(self,d_model):
        super(Net_atten, self).__init__()
        self.w2 = nn.Linear(25650, d_model)  # 25650
        self.layers_5 = TransformerEncoder( input_size=d_model,attention_heads= 4,num_blocks=5)
        self.layers_1 = TransformerEncoder(input_size=d_model,attention_heads= 1,num_blocks=1)
    def forward(self, input):
        corr = compute_complex_covariance4(input)
        corr=torch.view_as_real(corr)#B,T,F,C,C,2
        B, T, Frq, NUM_MIC, NUM_MIC,complex=corr.shape
        corr = corr.reshape(B, T, Frq *NUM_MIC*NUM_MIC,complex)
        corr = torch.cat([corr [..., 0], corr [..., 1]], 2)
        x2 = corr
        x=self.w2(x2)
        B,T,_=x.shape
        ilens = torch.tensor([T])
        x,olens,_,atten_4head=self.layers_5.forward(x, ilens)
        x,olens,_,atten_1head=self.layers_1.forward(x, ilens)
        atten = atten_1head.squeeze(1)
        corr = atten  @ corr
        part1, part2 = torch.split(corr, Frq *NUM_MIC*NUM_MIC, dim=2)
        part1 = torch.unsqueeze(part1, dim=3)
        part2 = torch.unsqueeze(part2, dim=3)
        corr= torch.cat([part1, part2], dim=3)
        corr = corr.reshape(B, T, Frq, NUM_MIC, NUM_MIC,complex)
        ########################################################
        return corr

class mvdr_atten(nn.Module):
    def __init__(self,d_model):
        super(mvdr_atten, self).__init__()
        self.STFT = STFT(1024, 256)
        self.FFT_SIZE=513
        self.net_work_speech = Net_atten(d_model)
        self.net_work_noise = Net_atten(d_model)
    # def forward(self, mixture,mask_speech,mask_noise):
    def forward(self, mixture, mask_speech):
        # mask_speech = torch.randn((1, 65, 513), dtype=torch.float32)
        [batch_size, _, channel] = mixture.shape  # (B,T,c)
        # if torch.cuda.is_available():
        #     mixture = mixture.cuda()  # (B*c,T)
        mixture = mixture.permute([0, 2, 1]).reshape(batch_size * channel, -1)  # (B*c,T)
        input_r, input_i = self.STFT.stft(mixture)  # (#(B*c,T,f,2)
        mix_spec = torch.stack([input_r, input_i], -1)
        mix_spec = mix_spec.reshape(batch_size, channel, -1, self.FFT_SIZE, 2)  # (B,c,T,f,2)
        mix_spec = mix_spec.permute(0, 2, 3, 1, 4)  # (B,T,f,c,2)

        ####################################################
        mix_real = mix_spec[..., 0]
        mix_imag = mix_spec[..., 1]
        mix_magnitude = torch.sqrt(mix_real ** 2 + mix_imag ** 2)
        mix_phase = torch.atan2(mix_imag, mix_real)
        mix_magnitude_mask = mix_magnitude * mask_speech.unsqueeze(-1)
        est_real = mix_magnitude_mask * torch.cos(mix_phase)
        est_imag = mix_magnitude_mask * torch.sin(mix_phase)
        est_speech = torch.stack([est_real, est_imag], -1)  # B,Frq,T,C,2
        est_noise = mix_spec - est_speech  # B,Frq,T,C,2
        ######################################################

        corr_speech = self.net_work_speech(est_speech)
        corr_noise = self.net_work_noise(est_noise)
        corr_speech = torch.view_as_complex(corr_speech)
        corr_noise = torch.view_as_complex(corr_noise)
        est=mvdr(mix_spec,corr_speech,corr_noise)
        est=est.permute(0,3,2,1)
        est_speech = self.STFT.istft(est)
        return est_speech

def mvdr(spec_mix_t,corr_speech,corr_noise):
    Phi_N = corr_noise.permute(0,2,1,3,4)
    Phi_S = corr_speech.permute(0,2,1,3,4)
    device = Phi_S.device
    u1 = torch.tensor([1, 0, 0, 0, 0], dtype=torch.cfloat).to(device)
    B, F, T, _, _ = Phi_S.shape
    # Phi_N_inv = complex_number_inverse(Phi_N)
    epsilon = 1e-7
    eye_complex = torch.eye(5, device=Phi_N.device) + 1j * torch.eye(5, device=Phi_N.device)
    eye_complex = eye_complex.expand_as(Phi_N)
    Phi_N = Phi_N + epsilon * eye_complex
    Phi_N_inv = torch.inverse(Phi_N)
    # Phi_N_inv=complex_matrix_inverse(Phi_N.real,Phi_N.imag)
    Phi_N_inv_Phi_S = complex_multi(Phi_N_inv, Phi_S)
    trace_Phi_N_inv_Phi_S = torch.einsum('...ii->...', Phi_N_inv_Phi_S)
    # trace_Phi_N_inv_Phi_S=trace_of_complex_matrix(Phi_N_inv_Phi_S)
    u1 = u1.expand(B, F, T, 5).unsqueeze(-1)
    a1 = complex_multi(Phi_N_inv_Phi_S, u1)  # molecule
    # 获取数据类型的最小正数 epsilon
    epsilon_value = torch.finfo(trace_Phi_N_inv_Phi_S.dtype).eps
    # 创建一个复数，其实部和虚部都是 epsilon_value
    epsilon = torch.tensor(epsilon_value) + 1j * torch.tensor(epsilon_value)
    W_t_f = (a1) / (trace_Phi_N_inv_Phi_S.unsqueeze(-1).unsqueeze(-1) + epsilon)
    W_t_f = W_t_f.permute(0, 1, 2, 4, 3)
    W_t_f_H = torch.conj(W_t_f)
    stft_mix = torch.view_as_complex(torch.stack([spec_mix_t[..., 0], spec_mix_t[..., 1]], -1)).unsqueeze(-1)
    stft_mix=stft_mix.permute(0,2,1,3,4)
    S_hat_t_f = complex_multi(W_t_f_H, stft_mix).squeeze(-1).squeeze(-1)
    est = torch.stack([S_hat_t_f.real, S_hat_t_f.imag], dim=-1)
    return est




if __name__ == '__main__':
    x = torch.randn((1, 16000, 5), dtype=torch.float32)#B*F*T*C*2
    mask = torch.randn((1, 65, 513), dtype=torch.float32)
    d_model=256
    net_work = mvdr_atten(d_model)
    print("ConvTasNet #param: {:.2f}".format(param( net_work)))
    # est=net_work(x,mask,mask)
    est = net_work(x, mask)

    # from ptflops import get_model_complexity_info
    # with torch.no_grad():
    #     # 获取模型的计算复杂度信息
    #     macs, params = get_model_complexity_info(
    #         net_work, (16000,5), as_strings=False, print_per_layer_stat=True, verbose=False
    #     )
    # macs = macs/1e9
    # params = params/1e6
    # # 打印 MACs 和参数量
    # print(f"MACs: {macs}")
    # print(f"Params: {params}")
