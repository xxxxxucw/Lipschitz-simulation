import numpy as np
import math
import pandas as pd
import os
import multiprocessing
import shutil
from tqdm import tqdm  # 引入进度条库

def time_series_func(x):
    """
    模拟R代码中的信号结构：
    一维时间序列，前半段频率为 10Hz，后半段频率为 20Hz (模拟变点)。
    x: time points in [0, 1]
    """
    f1 = 10.0
    f2 = 20.0
    
    y = np.zeros_like(x)
    mask = x <= 0.5
    
    y[mask] = np.sin(2 * math.pi * f1 * x[mask])
    # 保持相位连续性
    phase_shift = np.sin(2 * math.pi * f1 * 0.5) - np.sin(2 * math.pi * f2 * 0.5)
    y[~mask] = np.sin(2 * math.pi * f2 * x[~mask]) 
    
    return y

def save_data_to_csv(x, y, filename, n, m):
    """保存为标准CSV格式"""
    df = pd.DataFrame()
    df['sample_id'] = np.arange(len(x))
    df['measurement_id'] = 0 
    df['feature_1'] = x 
    df['target'] = y    
    df.to_csv(filename, index=False)

def generate_and_save(seed):
    """生成不同 n 的数据集"""
    # === [Check Point] 简单检查: 如果最后一个文件存在，假设该seed已完成 ===
    # 注意：这里我们用最大的n来做标记，如果想更严谨可以检查所有n的文件
    # 但通常数据生成很快，出错概率小。
    max_n = 12800 
    check_file = f"./data/train_data/n{max_n}m1s{seed}.csv"
    if os.path.exists(check_file):
        return #以此跳过已生成的seed

    np.random.seed(seed)
    n_vector = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    m_train = 1 
    
    os.makedirs(f"./data/train_data", exist_ok=True)
    os.makedirs(f"./data/valid_data", exist_ok=True)
    os.makedirs(f"./data/test_data", exist_ok=True)

    # 1. 生成统一的测试集
    test_file = f"./data/test_data/s{seed}.csv"
    if not os.path.exists(test_file):
        x_test = np.linspace(0, 1, 10000)
        y_test = time_series_func(x_test)
        save_data_to_csv(x_test, y_test, test_file, 10000, 1)

    # 2. 生成不同 n 的训练集和验证集
    for n in n_vector:
        train_file = f"./data/train_data/n{n}m{m_train}s{seed}.csv"
        valid_file = f"./data/valid_data/n{n}m{m_train}s{seed}.csv"
        
        # 只有文件不存在时才生成
        if not os.path.exists(train_file):
            x_train = np.sort(np.random.uniform(0, 1, n))
            y_train = time_series_func(x_train) + np.random.normal(0, 0.1, n)
            save_data_to_csv(x_train, y_train, train_file, n, m_train)
        
        if not os.path.exists(valid_file):
            n_valid = int(n * 0.25)
            x_valid = np.sort(np.random.uniform(0, 1, n_valid))
            y_valid = time_series_func(x_valid) + np.random.normal(0, 0.1, n_valid)
            save_data_to_csv(x_valid, y_valid, valid_file, n_valid, m_train)

if __name__ == '__main__':
    seeds = list(range(50)) 
    nproc = 16 
    
    # === [Modification] 不要删除旧数据，以便断点续传 ===
    # if os.path.exists("./data"):
    #     shutil.rmtree("./data")
    
    print(f"Start generating data with {nproc} processes...")
    with multiprocessing.Pool(processes=nproc) as pool:
        # 使用 tqdm 显示进度条
        # imap_unordered 比 map 更适合配合 tqdm 使用
        list(tqdm(pool.imap_unordered(generate_and_save, seeds), total=len(seeds), desc="Data Generation"))
    
    print("Data generation completed.")