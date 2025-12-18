import numpy as np
import math
import torch
import random 
import multiprocessing
import pandas as pd
import os

# ==========================================
# R 代码函数移植 (Functions f1-f5 from R)
# ==========================================

def f1(x):
    """ f1: Product interaction """
    return x[0] * x[1] * x[2]

def f2(x):
    """ f2: Mean """
    return np.mean(x)

def f3(x):
    """ f3: Sigmoid-like interaction """
    return 1 / (1 + np.exp(-3 * np.sum(x**2)))

def f4(x):
    """ f4: Log max """
    return np.log(1 + np.max(x))

def f5(x):
    """ f5: Exponential interaction """
    return np.exp(-np.min(x) - np.sqrt(x[0]) - np.sqrt(x[1]) - np.sqrt(x[2]))

# 选择要使用的函数
CURRENT_FUNC = f1
FUNC_NAME = "f1"

def mean_fun_tensor(x):
    return CURRENT_FUNC(x)

def generate_fun_tensor(n, m, signal_level=5, sigma=0.5):  
    """
    生成训练/验证数据
    """
    x_dim = 3
    x_base = np.random.uniform(low=0, high=1, size=(n, x_dim))
    x = np.repeat(x_base[:, np.newaxis, :], m, axis=1) # (n, m, 3)
    
    y = np.zeros((n, m))
    
    # 计算原始信号
    raw_signals = np.apply_along_axis(mean_fun_tensor, 2, x) # (n, m)
    
    # === 关键修正：归一化逻辑 ===
    # R代码逻辑：signal = signal_level * signal / sqrt(mean(signal^2))
    rms = np.sqrt(np.mean(raw_signals**2))
    if rms > 1e-8:
        clean_signal = signal_level * raw_signals / rms
    else:
        clean_signal = raw_signals

    # 添加噪声
    noise = np.random.normal(loc=0, scale=sigma, size=(n, m))
    y = clean_signal + noise
    
    return x, y

def save_3d_data_to_csv(x_3d, y, filename, n, m, d):
    """将3D数据保存为CSV格式"""
    data_list = []
    for sample_idx in range(n):
        for measurement_idx in range(m):
            row = [sample_idx, measurement_idx]
            row.extend(x_3d[sample_idx, measurement_idx, :])
            row.append(y[sample_idx, measurement_idx])
            data_list.append(row)
    
    feature_columns = [f'feature_{i+1}' for i in range(d)]
    columns =  ['sample_id', 'measurement_id'] + feature_columns + ['target']
    
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(filename, index=False)

def savedata(seed):
    """
    生成并保存数据
    """
    seed2 = ((seed+50) * 20000331) % 2**31
    torch.manual_seed(seed2) 
    np.random.seed(seed2) 
    random.seed(seed2) 
    
    os.makedirs("./data/train_data", exist_ok=True)
    os.makedirs("./data/valid_data", exist_ok=True)
    os.makedirs("./data/test_data", exist_ok=True)

    n_train_values = [10, 13, 17, 23, 31, 42, 57, 78, 106, 145, 198, 271, 400]
    m_train_values = [20] 
    
    # 信号强度参数 (保持与训练集一致)
    SIGNAL_LEVEL = 5

    # ==========================================
    # 关键修正：生成并归一化测试数据
    # ==========================================
    print(f"Seed {seed}: Generating Test Data...")
    x_test_base = np.random.uniform(low=0, high=1, size=(10000, 3))
    
    # 1. 计算原始测试信号
    raw_y_test = np.apply_along_axis(mean_fun_tensor, 1, x_test_base)
    
    # 2. 计算RMS并归一化 (确保测试集和训练集在同一尺度)
    rms_test = np.sqrt(np.mean(raw_y_test**2))
    if rms_test > 1e-8:
        y_test = SIGNAL_LEVEL * raw_y_test / rms_test
    else:
        y_test = raw_y_test
    
    # 保存测试集
    test_data = pd.DataFrame(x_test_base, columns=[f'feature_{i+1}' for i in range(3)])
    test_data['target'] = y_test
    test_data.to_csv(f"./data/test_data/s{seed}.csv", index=False)
    
    print(f"Seed {seed}: Generating Train/Valid Data...")
    for n_train in n_train_values:
        for m_train in m_train_values:
            n_valid = math.ceil(n_train * 0.25) 
            m_valid = m_train 
            
            # 生成训练数据 (内部已包含归一化逻辑)
            x, y = generate_fun_tensor(n=n_train, m=m_train, signal_level=SIGNAL_LEVEL) 
            
            # 生成验证数据
            x_valid, y_valid = generate_fun_tensor(n=n_valid, m=m_valid, signal_level=SIGNAL_LEVEL) 
            
            save_3d_data_to_csv(x, y, f"./data/train_data/n{n_train}m{m_train}s{seed}.csv",
                               n_train, m_train, 3)

            save_3d_data_to_csv(x_valid, y_valid, f"./data/valid_data/n{n_train}m{m_valid}s{seed}.csv",
                               n_valid, m_valid, 3)
    
    print(f"Seed {seed}: Completed.")

seedlist = list(range(50)) 

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    seeds = list(seedlist) 
    nproc = 20 
    with multiprocessing.Pool(processes = nproc) as pool: 
        pool.map(savedata, seedlist)