
# Attention-Based Beamformer For Multi-Channel Speech Enhancement
📄 **[Official Paper on IEEE Xplore](https://ieeexplore.ieee.org/iel8/10887540/10887541/10890720.pdf)**

## Abstract
Minimum Variance Distortionless Response (MVDR) is a classical adaptive beamformer that theoretically ensures the distortionless transmission of signals in the target direction, which makes it popular in real applications. Its noise reduction performance actually depends on the accuracy of the noise and speech spatial covariance matrices (SCMs) estimation. Time-frequency masks are often used to compute these SCMs. However, most mask-based beamforming methods typically assume that the sources are stationary, ignoring the case of
 moving sources, which leads to performance degradation. In this paper, we propose an attention-based mechanism to calculate the speech and noise SCMs and then apply MVDR to obtain the enhanced speech. To fully incorporate spatial information, the inplace convolution operator and frequency-independent LSTM are applied to facilitate SCMs estimation. The model is optimized in an end-to-end manner. Experiments demonstrate that the proposed method outperforms baselines with reduced computation and fewer parameters under various conditions.
<img width="1551" height="673" alt="image" src="https://github.com/user-attachments/assets/6b7082f8-47f9-44b3-a765-f96e96cd1011" />


 ## Performance
<p align="center">
 Table 1: STOI[%], ESTOI[%], PESQ, SI-SDR, WER[%] and TSOS[%] for non-moving and moving datasets
 <img width="1174" height="192" alt="image" src="https://github.com/user-attachments/assets/62fb54d0-f52e-4193-b731-069685ff659b" />
</p>
 
<p align="center">
 Table 2: Ablation experiments of ABIC-MVDR on moving datasets
  <img src="https://github.com/user-attachments/assets/4902f89b-c560-4f23-ac56-393660a2cc60" width="600"/>
</p>


 ## Usage

To train our proposed model, run the following command:

```bash
python ICRN_mask_mvdr.py
```

ATT_MVDR：

```bash
python conv_tasnet.py 
python net_atten.py
```

BLOCK-MVDR
```bash
python ICRN_mask.py
python eval_blockwize_MVDR.py
```

ONLINE-MVDR
```bash
python ICRN_mask.py
python eval_online_MVDR.py
```

CRN-MVDR
```bash
python CRN_mask_mvdr.py
```

## Regarding the reproduction of ATT-MVDR instructions.
📄 **[Official Paper on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10017367/)**

ATT-MVDR was reproduced as the baseline by us, with parameters aligned to original paper. We would like to thank Tsubasa Ochiai for his guidance in our reproduction work. 

 
## Acknowledgment
This research was supported by the China National Nature Science Foundation (No. 61876214). This work was also supported by the Open Fund (KF-2022-07-009) of Key Laboratory of Urban Land Resources Monitoring and Simulation, Ministry of Natural Resources, China.

 ## Citation
If you use ABIC-MVDR in your research or project, please cite the following paper:
```bash
@ARTICLE{10017367,
  author={Ochiai, Tsubasa and Delcroix, Marc and Nakatani, Tomohiro and Araki, Shoko},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Mask-Based Neural Beamforming for Moving Speakers With Self-Attention-Based Tracking}, 
  year={2023},
  volume={31},
  number={},
  pages={835-848},
  keywords={Array signal processing;Time-frequency analysis;Artificial neural networks;Microphone arrays;Filtering theory;Speech enhancement;Estimation;Array processing;mask-based neural beamformer;moving source;self-attention network;time-varying filter},
  doi={10.1109/TASLP.2023.3237172}}
```

