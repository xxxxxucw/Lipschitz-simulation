import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import os

# 设置绘图风格
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=15)
rc('text', usetex=False)

# 颜色和线条风格定义
color_tuple = [
    '#ae1908',  # red
    '#ec813b',  # orange
    '#05348b',  # dark blue
    '#9acdc4',  # pain blue
    '#00FF00',  # green
    '#800080'   # purple
]

lines = ['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed']
markers = ['o', 's', '^', 'v', 'D', 'x']

model_names = [
    'W=32, L=2', 
    'W=64, L=3', 
    'W=128, L=4', 
    'W=256, L=5', 
    'W=512, L=6', 
    'W=512, L=8'
]

def load_and_aggregate_results():
    """读取 ./perf/ 下的所有 CSV 并聚合结果"""
    
    n_vector = [100, 200, 400, 800, 1600, 3200]
    fixed_m = 50
    num_seeds = 10 
    num_models = 6

    # 初始化存储矩阵
    train_mse_mean = np.zeros((len(n_vector), num_models))
    test_mse_mean = np.zeros((len(n_vector), num_models))
    lip_mean = np.zeros((len(n_vector), num_models))
    
    missing_data = False

    for i, n in enumerate(n_vector):
        df_list = []
        for s in range(num_seeds):
            file_path = f"./perf/n{n}m{fixed_m}s{s}.csv"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df_list.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                pass

        if not df_list:
            print(f"Warning: No data found for n={n}")
            missing_data = True
            continue

        all_seeds_df = pd.concat(df_list, ignore_index=True)
        grouped = all_seeds_df.groupby('net_id').mean()
        
        for net_id in range(num_models):
            if net_id in grouped.index:
                train_mse_mean[i, net_id] = grouped.loc[net_id, 'train_loss']
                test_mse_mean[i, net_id] = grouped.loc[net_id, 'test_loss']
                lip_mean[i, net_id] = grouped.loc[net_id, 'lipschitz']
            else:
                lip_mean[i, net_id] = np.nan

    return n_vector, train_mse_mean, test_mse_mean, lip_mean, missing_data

def plot_results(n_vector, train_mse, test_mse, lip_mean):
    """绘制三张子图"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.3, bottom=0.2)

    # 1. Training Error (Linear Scale)
    for i in range(6):
        ax1.plot(n_vector, train_mse[:, i], color=color_tuple[i], linestyle=lines[i],
                 label=model_names[i], marker=markers[i], linewidth=2, markersize=6)
    ax1.set_xlabel(r"Sample Size $n$", fontsize=16)
    ax1.set_ylabel(r"Train MSE", fontsize=16)
    ax1.set_title(r"Training Error vs Sample Size", fontsize=18)
    ax1.set_xscale('log')
    # ax1.set_yscale('log')  <-- 已移除，默认线性
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(fontsize=10)

    # 2. Test Error (Linear Scale)
    for i in range(6):
        ax2.plot(n_vector, test_mse[:, i], color=color_tuple[i], linestyle=lines[i],
                 label=model_names[i], marker=markers[i], linewidth=2, markersize=6)
    ax2.set_xlabel(r"Sample Size $n$", fontsize=16)
    ax2.set_ylabel(r"Test MSE", fontsize=16)
    ax2.set_title(r"Test Error vs Sample Size", fontsize=18)
    ax2.set_xscale('log')
    # ax2.set_yscale('log') <-- 已移除，默认线性
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(fontsize=10)

    # 3. Lipschitz Constant (Log10 Scale as requested)
    # 取 Log10 展示
    log_lip = np.log10(lip_mean + 1e-8)
    
    for i in range(6):
        ax3.plot(n_vector, log_lip[:, i], color=color_tuple[i], linestyle=lines[i],
                 label=model_names[i], marker=markers[i], linewidth=2, markersize=6)
    
    ax3.set_xlabel(r"Sample Size $n$", fontsize=16)
    ax3.set_ylabel(r"Log10(Lipschitz Constant)", fontsize=16)
    ax3.set_title(r"Lipschitz Constant Growth", fontsize=18)
    ax3.set_xscale('log')
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend(fontsize=10)

    # 保存图片
    output_dir = "./Simulation/Case_GMM/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "GMM_Lipschitz_Trend_LinearLoss.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    print("Loading and plotting results...")
    n_vec, train_res, test_res, lip_res, missing = load_and_aggregate_results()
    
    if missing:
        print("Warning: Some data missing, plots might be incomplete.")
    
    if np.any(lip_res > 0):
        plot_results(n_vec, train_res, test_res, lip_res)
        print("Done.")
    else:
        print("No valid data found to plot.")