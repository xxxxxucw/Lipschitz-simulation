import numpy as np
import math
import torch
import random 
import multiprocessing
import pandas as pd
import os

# ==========================================
# 核心逻辑移植：对应 DenSimu.R 的数据生成
# ==========================================
def generate_densimu_data_batch(n, m, seed):
    """
    Python implementation of the simulation logic from DenSimu.R
    
    Parameters:
    n (int): Number of samples (subjects).
    m (int): Number of observations per sample (N in R code).
    """
    np.random.seed(seed)
    
    # 1. Generate Predictors X (9 dimensions as per DenSimu.R)
    # X1 ~ U(-1, 0), X2 ~ U(0, 1), X3 ~ U(1, 2)
    # X4 ~ N(0, 1), X5 ~ N(-10, 3), X6 ~ N(10, 3)
    # X7, X8, X9 ~ Bernoulli with p=[0.4, 0.3, 0.6]
    
    X1 = np.random.uniform(-1, 0, n)
    X2 = np.random.uniform(0, 1, n)
    X3 = np.random.uniform(1, 2, n)
    X4 = np.random.normal(0, 1, n)
    X5 = np.random.normal(-10, 3, n)
    X6 = np.random.normal(10, 3, n)
    X7 = np.random.choice([0, 1], n, p=[0.4, 0.6]) # Note: p is probability of 0, 1 in np.choice logic needs checking
    # R: sample(c(0,1), n, TRUE, c(0.4, 0.6)) means 0 with prob 0.4.
    X8 = np.random.choice([0, 1], n, p=[0.3, 0.7])
    X9 = np.random.choice([0, 1], n, p=[0.6, 0.4])

    # Stack features: shape (n, 9)
    X = np.stack([X1, X2, X3, X4, X5, X6, X7, X8, X9], axis=1)
    
    # Parameters from R code
    sigma0 = 3
    kappa = 1
    nu1 = 1 # mean parameter for normal distribution of mu
    
    # Initialize output array
    # Shape: (n, m, x_dim) and (n, m)
    x_expanded = np.zeros((n, m, 9))
    y = np.zeros((n, m))
    
    for i in range(n):
        # Extract features for i-th sample
        x_i = X[i]
        
        # 2. Calculate Expected Mean (Eta) and Sigma based on R formulas
        # expect_eta_Z = 3 * (sin(pi * x[i,1])+cos(pi * x[i,2])) * x[i,8] + (5*x[i,4]^2 + x[i,5]) * x[i,7]
        term1 = 3 * (np.sin(np.pi * x_i[0]) + np.cos(np.pi * x_i[1])) * x_i[7]
        term2 = (5 * (x_i[3]**2) + x_i[4]) * x_i[6]
        expect_eta_Z = term1 + term2
        
        # expect_sigma_Z = sigma0 + 0.5*(sin(pi * x[i,1])+cos(pi * x[i,2])) * x[i,8] + abs(5*x[i,4]^2 + x[i,5]) * x[i,7]
        term3 = 0.5 * (np.sin(np.pi * x_i[0]) + np.cos(np.pi * x_i[1])) * x_i[7]
        term4 = np.abs(5 * (x_i[3]**2) + x_i[4]) * x_i[6]
        expect_sigma_Z = sigma0 + term3 + term4
        
        # 3. Sample latent mean (mu) and sigma
        # mu = rnorm(1, mean = expect_eta_Z, sd = nu1) (Assuming nu1=1 from DenSimu.R)
        mu = np.random.normal(expect_eta_Z, 1.0) 
        
        # sigma = rgamma(1, shape = expect_sigma_Z^2/kappa, scale = kappa/expect_sigma_Z)
        # NumPy gamma uses shape (k) and scale (theta). 
        shape_param = (expect_sigma_Z**2) / kappa
        scale_param = kappa / expect_sigma_Z
        sigma = np.random.gamma(shape_param, scale_param)
        
        # 4. Generate m observations
        # y[[i]] = sort(mu + sigma * rnorm(N)) 
        # Note: We don't sort here because in regression we usually treat them as independent measurements in the batch
        y_i = mu + sigma * np.random.normal(0, 1, m)
        
        y[i, :] = y_i
        x_expanded[i, :, :] = x_i # Feature is static across m measurements for the same subject
        
    return x_expanded, y


def save_3d_data_to_csv(x_3d, y, filename, n, m, d):
    """
    保存为CSV，与你之前的逻辑一致
    """
    data_list = []
    
    # 优化：使用列表推导式或numpy操作加速，但为了保持逻辑清晰，这里使用循环
    # 为了加速，我们可以直接构建大数组然后转DataFrame
    
    # 构建 ID 列
    sample_ids = np.repeat(np.arange(n), m)
    meas_ids = np.tile(np.arange(m), n)
    
    # 重塑 X 和 Y
    x_flat = x_3d.reshape(n * m, d)
    y_flat = y.reshape(n * m)
    
    # 合并
    data_matrix = np.column_stack((sample_ids, meas_ids, x_flat, y_flat))
    
    # 创建列名
    feature_columns = [f'feature_{i+1}' for i in range(d)]
    columns =  ['sample_id', 'measurement_id'] + feature_columns + ['target']
    
    # 保存
    df = pd.DataFrame(data_matrix, columns=columns)
    
    # 确保ID是整数
    df['sample_id'] = df['sample_id'].astype(int)
    df['measurement_id'] = df['measurement_id'].astype(int)
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)


def savedata(seed):
    """
    Worker function
    """
    # 这里的种子生成逻辑保持你原有的风格
    seed2 = ((seed+50) * 20000331)%2**31
    np.random.seed(seed2) 
    
    # 你的 visualize.py 中定义的 n_vector 和 m_vector 需要被覆盖
    # R代码中的 n 为 [100, 200, 500, 1000]，我们这里为了兼容 visualize.py，
    # 建议使用你 visualize.py 里定义的列表，或者根据需要修改。
    # 此处使用 visualize.py 中的列表以保证画图兼容。
    n_train_values = [10, 13, 17, 23, 31, 42, 57, 78, 106, 145, 198, 271, 400]
    # 如果你想完全复刻 R 代码的 sample size，可以改成: [100, 200, 500, 1000]
    
    m_train_values = [20] # 固定 m=20 用于 visualize
    
    # 特征维度，DenSimu.R 中是 9
    d_feature = 9

    # 1. 生成测试集 (一次性生成)
    # x_test 需要是 (N_test, d)
    # R代码中 xout 也是同样的分布
    n_test = 10000
    m_test = 1 # 测试集通常看作单次观测或平均
    x_test, y_test = generate_densimu_data_batch(n_test, m_test, seed2)
    
    # 保存测试集
    # reshape x_test to (n_test, d)
    x_test_flat = x_test[:, 0, :]
    y_test_flat = y_test[:, 0]
    
    test_df = pd.DataFrame(x_test_flat, columns=[f'feature_{i+1}' for i in range(d_feature)])
    test_df['target'] = y_test_flat
    os.makedirs("./data/test_data/", exist_ok=True)
    test_df.to_csv(f"./data/test_data/s{seed}.csv", index=False)
    
    # 2. 生成训练和验证集
    for n_train in n_train_values:
        for m_train in m_train_values:
            n_valid = math.ceil(n_train * 0.25)
            m_valid = m_train
            
            # 生成训练数据
            x, y = generate_densimu_data_batch(n_train, m_train, seed2)
            
            # 生成验证数据
            x_valid, y_valid = generate_densimu_data_batch(n_valid, m_valid, seed2 + 1) # shift seed for valid
            
            # 保存
            save_3d_data_to_csv(x, y, f"./data/train_data/n{n_train}m{m_train}s{seed}.csv",
                               n_train, m_train, d_feature)

            save_3d_data_to_csv(x_valid, y_valid, f"./data/valid_data/n{n_train}m{m_valid}s{seed}.csv",
                               n_valid, m_valid, d_feature)
    
    print(f"Seed {seed} completed.")

# ==========================================
# 主执行块
# ==========================================
seedlist = list(range(50)) 

# 4090 并行生成数据完全没问题，CPU负责计算，硬盘负责写入
# spawn 模式更安全
multiprocessing.set_start_method('spawn', force=True)

if __name__ == '__main__':
    seeds = list(seedlist)
    nproc = 20 # 根据CPU核数调整，数据生成主要耗费CPU
    
    print("Starting data generation based on DenSimu.R logic...")
    with multiprocessing.Pool(processes = nproc) as pool: 
        pool.map(savedata, seedlist)
    print("All data generated.")