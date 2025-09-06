# main.py

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# 导入您自己的模块
from sic import sic_hybrid_precoding_ofdm
from codebook import gen_upa_cb
from swomp import estimate_swomp, recon_chan
from util import getChannel, calc_rate, gen_steer_vec
from quantization import AGQuantizer


def main():
    # --- 仿真参数 ---
    Nt = 64
    Nr = 1
    K = 2  # 用户数
    Ns = K
    NRF = K
    Nc = 32
    L = 2  # 信道路径数
    Q = 8  # 导频长度
    nt_dims = (8, 8)
    SNR_DB = 10
    snr = 10 ** (SNR_DB / 10)
    # 反馈比特数B的变化范围
    B_vals = [1, 3, 16, 24, 32, 48, 64]

    # --- 数据集和码本加载 ---
    try:
        data = loadmat("../data/H_UPA_4.mat")
        print("数据集 'H_UPA_4.mat' 加载成功。")
    except FileNotFoundError:
        print("错误: 数据集 'H_UPA_4.mat' 未在 '../data/' 目录下找到。")
        return

    H_data = data['H_UPA']
    H_data = H_data[:, :K, :, :]
    N_BATCH = H_data.shape[0]

    omp_ang_n = 16
    At = gen_upa_cb(nt_dims, omp_ang_n)
    Xp = (1 / np.sqrt(Q * Nt)) * (np.random.randn(Q, Nt) + 1j * np.random.randn(Q, Nt))

    # --- 用于量化器训练的随机数据 ---
    train_g = (np.random.randn(1000, K, L) + 1j * np.random.randn(1000, K, L)) / np.sqrt(2)
    train_phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=10000)
    train_theta = np.random.uniform(0, np.pi, size=10000)

    # --- 结果存储 ---
    rates_ideal_digital = np.zeros(N_BATCH)
    rates_sic_perfect = np.zeros(N_BATCH)
    rates_sic_estimated = np.zeros(N_BATCH)
    # 为随B变化的结果创建一个二维数组
    rates_sic_quantized_vs_B = np.zeros((len(B_vals), N_BATCH))

    # --- 主仿真循环 ---
    for si in range(N_BATCH):
        print(f"处理样本: {si + 1}/{N_BATCH}")

        H_user = H_data[si, :, :, :]
        H_sys = H_user.reshape(K * Nr, Nc, Nt)

        # --- 基准 1: 理想全数字 (完美CSI) ---
        Fopt_p, Wopt_p = getChannel(H_sys, K, Nr, Ns)
        rate_p_temp = 0.0
        for k in range(Nc):
            Fk, Wk, Hk = Fopt_p[:, :, k], Wopt_p[:, :, k], H_sys[:, k, :]
            H_eff = Wk.conj().T @ Hk @ Fk
            det_term = np.eye(Ns) + (snr / Ns) * (H_eff @ H_eff.conj().T)
            sign, logdet = np.linalg.slogdet(det_term)
            if sign > 0: rate_p_temp += logdet / np.log(2)
        rates_ideal_digital[si] = np.real(rate_p_temp) / Nc

        # --- 基准 2: SIC (完美CSI) ---
        FRF_p, FBB_p = sic_hybrid_precoding_ofdm(H_sys, NRF, Ns, At)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_p @ FBB_p[:, :, k], 'fro')
            if norm_f > 1e-9: FBB_p[:, :, k] *= (np.sqrt(Ns) / norm_f)
        rates_sic_perfect[si] = calc_rate(H_user, FRF_p, FBB_p, Wopt_p, snr, K, Ns, Nc)

        # --- 信道估计 (所有后续步骤的基础) ---
        Y_clean = np.tensordot(H_sys, Xp.T, axes=([2], [0]))
        n_var = 1 / snr
        noise = np.sqrt(n_var / 2) * (np.random.randn(*Y_clean.shape) + 1j * np.random.randn(*Y_clean.shape))
        Y = Y_clean + noise
        g, phi, theta, _ = estimate_swomp(Y.reshape(K, Nc, Q), Xp, At, L, omp_ang_n)

        # --- 基准 3: SIC (仅估计CSI, 无量化) ---
        A_est = np.hstack([gen_steer_vec(nt_dims, phi[l], theta[l]) for l in range(L)])
        H_hat = recon_chan(A_est, g, L)
        H_hat_sys = H_hat.reshape(K * Nr, Nc, Nt)
        _, Wopt_e = getChannel(H_hat_sys, K, Nr, Ns)
        FRF_e, FBB_e = sic_hybrid_precoding_ofdm(H_hat_sys, NRF, Ns, At)
        for k in range(Nc):
            norm_f = np.linalg.norm(FRF_e @ FBB_e[:, :, k], 'fro')
            if norm_f > 1e-9: FBB_e[:, :, k] *= (np.sqrt(Ns) / norm_f)
        rates_sic_estimated[si] = calc_rate(H_user, FRF_e, FBB_e, Wopt_e, snr, K, Ns, Nc)

        # --- 4. SIC (估计CSI + 有限反馈) ---
        # 循环遍历不同的B值
        for b_idx, b_val in enumerate(B_vals):
            quantizer = AGQuantizer(b_val, L, K, train_g, train_phi, train_theta)
            g_avg = np.mean(g, axis=2)
            q_data = quantizer.quantize(g_avg, phi, theta)
            g_q_avg, phi_q, theta_q = quantizer.dequantize(q_data)

            g_q = np.tile(np.expand_dims(g_q_avg, axis=2), (1, 1, Nc))
            A_q = np.hstack([gen_steer_vec(nt_dims, phi_q[l], theta_q[l]) for l in range(L)])
            H_q = recon_chan(A_q, g_q, L)
            H_q_sys = H_q.reshape(K * Nr, Nc, Nt)

            _, Wopt_q = getChannel(H_q_sys, K, Nr, Ns)
            FRF_q, FBB_q = sic_hybrid_precoding_ofdm(H_q_sys, NRF, Ns, At)
            for k in range(Nc):
                norm_f = np.linalg.norm(FRF_q @ FBB_q[:, :, k], 'fro')
                if norm_f > 1e-9: FBB_q[:, :, k] *= (np.sqrt(Ns) / norm_f)

            rates_sic_quantized_vs_B[b_idx, si] = calc_rate(H_user, FRF_q, FBB_q, Wopt_q, snr, K, Ns, Nc)

    # --- 结果汇总与绘图 ---
    rate_ideal_avg = np.mean(rates_ideal_digital)
    rate_sic_perfect_avg = np.mean(rates_sic_perfect)
    rate_sic_estimated_avg = np.mean(rates_sic_estimated)
    rate_sic_quantized_avg = np.mean(rates_sic_quantized_vs_B, axis=1)

    print("\n" + "=" * 60)
    print(f"仿真完成 (SNR={SNR_DB} dB, {N_BATCH}个样本)")
    print("-" * 60)
    print(f"  - 基准1: 理想全数字 (完美CSI)             : {rate_ideal_avg:.4f} bits/s/Hz")
    print(f"  - 基准2: SIC (完美CSI)                    : {rate_sic_perfect_avg:.4f} bits/s/Hz")
    print(f"  - 基准3: SIC (估计CSI, 无量化)            : {rate_sic_estimated_avg:.4f} bits/s/Hz")
    print("\n--- 性能随反馈比特数B变化 ---")
    for b, rate in zip(B_vals, rate_sic_quantized_avg):
        print(f"  - B = {b:<3} bits, Rate: {rate:.4f} bits/s/Hz")
    print("=" * 60)

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.axhline(y=rate_ideal_avg, color='black', linestyle='-', label='Ideal Full-Digital (Perfect CSI)')
    plt.axhline(y=rate_sic_perfect_avg, color='blue', linestyle='--', label='SIC (Perfect CSI)')
    plt.axhline(y=rate_sic_estimated_avg, color='green', linestyle=':', label='SIC (Estimated CSI, No Quantization)')
    plt.plot(B_vals, rate_sic_quantized_avg, 'r-s', label='SIC (Estimated CSI + Limited Feedback)')

    plt.xscale('log', base=2)
    plt.grid(True, which='both')
    plt.xlabel('Total Feedback Bits (B)')
    plt.ylabel('Spectral Efficiency (bits/s/Hz)')
    plt.title(f'Performance vs. Feedback Bits (SIC Algorithm, SNR = {SNR_DB} dB)')
    plt.legend()
    plt.xticks(B_vals, labels=B_vals)
    plt.minorticks_off()
    plt.show()


if __name__ == '__main__':
    main()