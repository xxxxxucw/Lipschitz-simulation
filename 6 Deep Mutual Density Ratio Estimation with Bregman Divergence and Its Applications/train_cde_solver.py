import argparse
import time
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

# 引入原项目工具
from utils.basic_utils import setup_seed, weight_init, BasicFNN, accuracy_eval
from utils.estimators_utils import EnsembleEst

# 适配 4090
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for batch in self.dl:
            yield [x.to(self.device) for x in batch]
    def __len__(self):
        return len(self.dl)

def compute_lipschitz(model):
    """计算 BasicFNN 的全局 Lipschitz 常数"""
    lip = 1.0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                _, s, _ = torch.linalg.svd(w)
                lip *= s[0].item()
    return lip

def train_one_epoch_cde(n_train, seed, args, save_path_net):
    setup_seed(seed, False)
    
    # --- 1. 读取数据 ---
    try:
        train_df = pd.read_csv(f'./data/train/n{n_train}_s{seed}.csv')
        valid_df = pd.read_csv(f'./data/valid/n{n_train}_s{seed}.csv')
        test_df = pd.read_csv(f'./data/test/test_s{seed}.csv')
    except FileNotFoundError:
        return None

    feat_cols = [c for c in train_df.columns if 'feature_' in c]
    x_train = train_df[feat_cols].values
    y_train = train_df['target'].values.reshape(-1, 1)
    x_val = valid_df[feat_cols].values
    y_val = valid_df['target'].values.reshape(-1, 1)
    
    x_test = test_df[feat_cols].values
    y_test = test_df['target'].values.reshape(-1, 1)
    cond_mean_test = test_df['cond_mean'].values.reshape(-1, 1)
    cond_std_test = test_df['cond_std'].values.reshape(-1, 1)
    
    test_n = x_test.shape[0]

    # --- 2. 恢复 KDE 计算 (用于评估) ---
    # 这是 CDE 方法评估所必需的
    y_bandwid = 1.63 * np.std(y_train) * (n_train ** (-1 / 3))
    y_kde = KernelDensity(kernel='gaussian', bandwidth=y_bandwid).fit(y_train)

    # --- 3. 准备 PyTorch 数据 ---
    train_dat = TensorDataset(torch.from_numpy(y_train).float(), torch.from_numpy(x_train).float())
    val_dat = TensorDataset(torch.from_numpy(y_val).float(), torch.from_numpy(x_val).float())

    train_loader_base = DataLoader(train_dat, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader_base = DataLoader(val_dat, batch_size=len(val_dat), shuffle=False, pin_memory=True)

    trainloader = DeviceDataLoader(train_loader_base, device)
    val_loader = DeviceDataLoader(val_loader_base, device)

    # --- 4. 初始化网络 ---
    x_dim = x_train.shape[1]
    widths_array = [[8, 8], [64, 64], [32, 32, 32]] 
    list_id = args.model_type - 1
    width_vec = [x_dim + 1,] + widths_array[list_id]
    
    mdr_net = BasicFNN(1, x_dim, width_vec).to(device)
    mdr_net.apply(weight_init)

    # --- 5. 训练 ---
    lr_milestones = [200, 400, 600] 
    optimizer = torch.optim.Adam(mdr_net.parameters(), lr=args.lr, weight_decay=args.wd)
    best_model_name = f"{save_path_net}/best_n{n_train}_s{seed}.pth"
    
    ensemble_est = EnsembleEst(
        loss_type=args.Loss_type, 
        sub_loss_type=args.Sub_loss_type, 
        model=mdr_net,
        optimizer=optimizer, 
        best_model=best_model_name, 
        args=args
    )
    
    ensemble_est.train_scheduler(train_loader=trainloader, val_loader=val_loader, lr_milestones=lr_milestones)

    # --- 6. 评估 (恢复原论文的积分逻辑) ---
    if os.path.exists(best_model_name):
        # 消除 FutureWarning
        checkpoint = torch.load(best_model_name, weights_only=False)
        mdr_net.load_state_dict(checkpoint['state_dict'])
        os.remove(best_model_name)
    
    mdr_net.eval()
    
    # 6.1 计算 Lipschitz
    lipschitz_val = compute_lipschitz(mdr_net)

    # 6.2 计算 CDE 误差 (Mean MSE 和 Std MSE)
    # 这部分代码直接移植自 cde_est_simulation.py
    n_grid = 1000
    y_grid = np.linspace(y_test.min(), y_test.max(), n_grid)[:-1].reshape((-1, 1))
    ty_test = torch.from_numpy(y_grid).float().to(device)
    log_density_y = y_kde.score_samples(y_grid).reshape((-1, 1))
    
    # 预计算一些常量
    one_vec = torch.ones((n_grid - 1, 1)).to(device)
    est_cmean_array = np.zeros((test_n, 1))
    est_cstd_array = np.zeros((test_n, 1))
    
    # 使用 no_grad 加速
    with torch.no_grad():
        for j in range(test_n):
            # 构造输入 (grid_y, current_x)
            tx_test = torch.from_numpy(x_test[j]).float().to(device) * one_vec
            
            # 网络预测 (Ratio)
            # mdr_net 输入是 (y, x)
            output = mdr_net(ty_test, tx_test).cpu().numpy()
            temp = np.exp(output)
            
            # 估计密度
            est_cde = temp * np.exp(log_density_y)
            
            # 积分计算均值
            est_cmean = (y_test.max() - y_test.min()) / (n_grid - 1) * np.sum(y_grid * est_cde)
            est_cmean_array[j] = est_cmean
            
            # 积分计算标准差
            est_cstd = np.sqrt((y_test.max() - y_test.min()) / (n_grid - 1) * np.sum((y_grid - est_cmean) ** 2 * est_cde))
            est_cstd_array[j] = est_cstd

    cmean_mse = accuracy_eval(est_cmean_array, cond_mean_test) ** 2
    cstd_mse = accuracy_eval(est_cstd_array, cond_std_test) ** 2

    return {
        'n_train': n_train,
        'seed': seed,
        'lipschitz': lipschitz_val,
        'mean_mse': cmean_mse,
        'std_mse': cstd_mse
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--epochs', default=800, type=int) 
    parser.add_argument('--model_type', default=1, type=int)
    parser.add_argument('--Loss_type', default='lrb', type=str)
    parser.add_argument('--Sub_loss_type', default='LR', type=str)
    parser.add_argument('--patience', default=100, type=int)
    args = parser.parse_args()

    result_dir = './perf'
    save_path_net = './temp_nets'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(save_path_net, exist_ok=True)
    
    n_train_values = [500, 1000, 2000, 3000, 5000, 8000, 10000]
    seeds = range(50)

    print(f"Starting CDE Estimation (Full Metrics) on {device}...")

    for n in n_train_values:
        result_file = f'{result_dir}/cde_res_n{n}.csv'
        
        if os.path.exists(result_file):
            print(f"Skipping n={n}, results exist.")
            continue
            
        results_list = []
        pbar = tqdm(seeds, desc=f"Training n={n}")
        
        for seed in pbar:
            res = train_one_epoch_cde(n, seed, args, save_path_net)
            if res:
                results_list.append(res)
                # 进度条显示 Lipschitz 和 Mean MSE
                pbar.set_postfix({'Lip': f"{res['lipschitz']:.2f}", 'MSE': f"{res['mean_mse']:.4f}"})
        
        if results_list:
            pd.DataFrame(results_list).to_csv(result_file, index=False)
            print(f"Saved results for n={n}")

    if os.path.exists(save_path_net) and not os.listdir(save_path_net):
        os.rmdir(save_path_net)

if __name__ == '__main__':
    main()