import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
# 确保 utils 在 Python 路径中
from utils.simulated_data_generating_utils import cde_sim_datasets 
from utils.basic_utils import setup_seed

def save_mixed_data_to_csv(x, y, filename):
    """
    保存数据以匹配 cde_est 的读取习惯
    原始代码是将 x 和 y 拼接保存，这里保持逻辑一致，但加上表头方便 pandas 读取
    """
    # y shape: (n, 1), x shape: (n, dim)
    data = np.hstack((x, y.reshape(-1, 1)))
    feature_cols = [f'feature_{i}' for i in range(x.shape[1])]
    cols = feature_cols + ['target']
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(filename, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=1, type=int) # 默认为模型1，可命令行修改
    parser.add_argument('--seed_start', default=0, type=int)
    parser.add_argument('--num_rep', default=50, type=int)
    args = parser.parse_args()

    model_type = args.model_type
    # 定义样本量梯度，用于观察 Lipschitz 变化
    # 根据原代码默认值，覆盖从中等到大的样本量
    n_train_values = [500, 1000, 2000, 3000, 5000, 8000, 10000]
    
    seeds = range(args.seed_start, args.seed_start + args.num_rep)
    
    base_path = './data'
    dirs = [f'{base_path}/train', f'{base_path}/valid', f'{base_path}/test']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print(f"Start Data Generation for CDE Model Type {model_type}...")
    
    # 1. 生成固定大小的测试集 (Test Set)
    # 原代码逻辑：test_n = 2000
    test_n = 2000
    for seed in tqdm(seeds, desc="Generating Test Data"):
        setup_seed(seed, False)
        test_file = f'{base_path}/test/test_s{seed}.csv'
        
        if not os.path.exists(test_file):
            y_test, x_test, cond_mean_test, cond_std_test = cde_sim_datasets(test_n, model_type)
            # 测试集还需要保存真实的 mean 和 std 用于评估
            # 格式: feature... target mean std
            data = np.hstack((x_test, y_test.reshape(-1, 1), 
                              cond_mean_test.reshape(-1, 1), cond_std_test.reshape(-1, 1)))
            cols = [f'feature_{i}' for i in range(x_test.shape[1])] + ['target', 'cond_mean', 'cond_std']
            pd.DataFrame(data, columns=cols).to_csv(test_file, index=False)

    # 2. 生成不同 n 的训练集和验证集
    for n in tqdm(n_train_values, desc="Generating Train/Valid Data"):
        for seed in seeds:
            setup_seed(seed, False)
            
            train_file = f'{base_path}/train/n{n}_s{seed}.csv'
            valid_file = f'{base_path}/valid/n{n}_s{seed}.csv'
            
            if os.path.exists(train_file) and os.path.exists(valid_file):
                continue
            
            # 训练集
            y_train, x_train, *_ = cde_sim_datasets(n, model_type)
            save_mixed_data_to_csv(x_train, y_train, train_file)
            
            # 验证集 (固定大小 2000)
            y_val, x_val, *_ = cde_sim_datasets(2000, model_type)
            save_mixed_data_to_csv(x_val, y_val, valid_file)

    print("Data generation completed.")

if __name__ == '__main__':
    main()