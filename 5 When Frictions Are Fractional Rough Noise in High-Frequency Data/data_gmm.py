import numpy as np
import pandas as pd
import multiprocessing
import os
from scipy.fft import fft, ifft
from tqdm import tqdm

def autocovariance_fgn(n, H):
    k = np.arange(n)
    gamma = 0.5 * (np.abs(k + 1)**(2 * H) + np.abs(k - 1)**(2 * H) - 2 * np.abs(k)**(2 * H))
    return gamma

def mixed_fgn_acf(n, H, sigma=0.02, nsr=1/3, N0=23400):
    if H == 0.5:
        rho = 0
    else:
        try:
            rho = np.sqrt(sigma**2 * N0**(2 * H - 1) * nsr / (1 - nsr))
        except:
            rho = 0
    acf_noise = autocovariance_fgn(n, 0.5)
    acf_signal = autocovariance_fgn(n, H)
    acf_total = (sigma**2) * acf_noise + (rho**2) * acf_signal
    return acf_total

def davies_harte_simulation(n, acf):
    M = 2 * n
    c = np.concatenate([acf, [0], acf[1:][::-1]])
    eigenvals = fft(c).real
    eigenvals[eigenvals < 0] = 0
    Z = np.random.randn(M) + 1j * np.random.randn(M)
    W = fft(np.sqrt(eigenvals) * Z)
    X = (W.real)[:n] / np.sqrt(M)
    return X

def generate_data_batch(n_samples, seq_len, seed):
    np.random.seed(seed)
    x_list = []
    y_list = [] 
    sigma = 0.02
    nsr = 1/3
    N0 = 23400 
    for i in range(n_samples):
        target_H = np.random.uniform(0.05, 0.95)
        acf = mixed_fgn_acf(seq_len, target_H, sigma, nsr, N0)
        dY = davies_harte_simulation(seq_len, acf)
        dY = (dY - np.mean(dY)) / (np.std(dY) + 1e-8)
        x_list.append(dY)
        y_list.append(target_H)
    return np.array(x_list), np.array(y_list)

def save_to_csv(x_data, y_data, filename):
    n_samples, seq_len = x_data.shape
    sample_ids = np.repeat(np.arange(n_samples), seq_len)
    measure_ids = np.tile(np.arange(seq_len), n_samples)
    features = x_data.flatten()
    targets = np.repeat(y_data, seq_len)
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'measurement_id': measure_ids,
        'feature_1': features,
        'target': targets
    })
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

def worker_task(seed):
    n_train_values = [100, 200, 400, 800, 1600, 3200]
    m_train_values = [50] 
    
    # [修正] 移除 if seed == 0 限制，确保每个 seed 都生成测试集
    test_file = f"./data/test_data/s{seed}.csv"
    if not os.path.exists(test_file):
        # 测试集大小固定 1000
        x_test, y_test = generate_data_batch(1000, m_train_values[0], seed * 999) 
        save_to_csv(x_test, y_test, test_file)

    for n_train in n_train_values:
        for m_train in m_train_values:
            train_file = f"./data/train_data/n{n_train}m{m_train}s{seed}.csv"
            valid_file = f"./data/valid_data/n{n_train}m{m_train}s{seed}.csv"

            if os.path.exists(train_file) and os.path.exists(valid_file):
                continue

            current_seed = seed * 10000 + n_train + m_train
            
            if not os.path.exists(train_file):
                x_train, y_train = generate_data_batch(n_train, m_train, current_seed)
                save_to_csv(x_train, y_train, train_file)
            
            if not os.path.exists(valid_file):
                n_valid = int(np.ceil(n_train * 0.25))
                x_valid, y_valid = generate_data_batch(n_valid, m_train, current_seed + 1)
                save_to_csv(x_valid, y_valid, valid_file)
    return f"Seed {seed} done"

if __name__ == '__main__':
    seeds = list(range(10)) 
    print(f"Fixing missing data for {len(seeds)} seeds...")
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=10) as pool:
        for _ in tqdm(pool.imap_unordered(worker_task, seeds), total=len(seeds), desc="Generating Data"):
            pass
    print("All data generation tasks completed.")