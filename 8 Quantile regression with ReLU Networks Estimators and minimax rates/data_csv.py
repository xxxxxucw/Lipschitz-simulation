import numpy as np
import math
import torch
import random 
import multiprocessing
import pandas as pd
import os
from funcs import Scenario1 # 引入目标论文中的数据生成场景

# 自动创建目录
for d in ['./data/train_data', './data/valid_data', './data/test_data']:
    os.makedirs(d, exist_ok=True)

def save_2d_data_to_csv(x_2d, y, filename, n, d):

    data_list = []
    for sample_idx in range(n):
        row = [n, 0]  # sample_id=n, measurement_id=0 (伪造的 m=1)
        row.extend(x_2d[sample_idx, :])  # 特征值
        row.append(y[sample_idx])        # 目标值
        data_list.append(row)
    
    feature_columns = [f'feature_{i+1}' for i in range(d)]
    columns = ['sample_id', 'measurement_id'] + feature_columns + ['target']
    
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(filename, index=False)

def savedata(seed):
    seed2 = ((seed+50) * 20000331)%2**31
    torch.manual_seed(seed2) 
    np.random.seed(seed2) 
    random.seed(seed2) 

    # 替换为你论文中需要的样本量梯度
    n_train_values = [100, 500, 1000, 2000, 5000, 10000]
    m_train = 1 
    
    # 实例化目标论文的数据生成器
    func = Scenario1()
    x_dim = func.n_in

    # 1. 生成统一的测试数据
    n_test = 10000
    x_test = np.random.uniform(low=0, high=1, size=(n_test, x_dim))
    y_test = func.sample(x_test)
    
    test_data = pd.DataFrame(x_test, columns=[f'feature_{i+1}' for i in range(x_dim)])
    test_data['target'] = y_test
    test_data.to_csv(f"./data/test_data/s{seed}.csv", index=False)
    
    # 2. 生成不同规模的训练集和验证集
    for n_train in n_train_values:
        n_valid = math.ceil(n_train * 0.25) 
        
        # 训练集
        x_train = np.random.uniform(low=0, high=1, size=(n_train, x_dim))
        y_train = func.sample(x_train)
        
        # 验证集
        x_valid = np.random.uniform(low=0, high=1, size=(n_valid, x_dim))
        y_valid = func.sample(x_valid)
        
        save_2d_data_to_csv(x_train, y_train, f"./data/train_data/n{n_train}m{m_train}s{seed}.csv", n_train, x_dim)
        save_2d_data_to_csv(x_valid, y_valid, f"./data/valid_data/n{n_train}m{m_train}s{seed}.csv", n_valid, x_dim)
        
    print(f"Seed {seed} 数据生成完毕！")

if __name__ == '__main__':
    seedlist = list(range(10)) # 先跑10个seed测试
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=10) as pool: 
        pool.map(savedata, seedlist)