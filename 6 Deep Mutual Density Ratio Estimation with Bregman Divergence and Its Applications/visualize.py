import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=15)

def main():
    result_path = './perf'
    all_files = glob.glob(os.path.join(result_path, "cde_res_n*.csv"))
    
    if not all_files:
        print("No result files found in ./perf/")
        return

    df_list = []
    for f in all_files:
        df_list.append(pd.read_csv(f))
    
    if not df_list:
        print("Empty data.")
        return

    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by='n_train')
    
    # 聚合数据
    n_values = df['n_train'].unique()
    n_values.sort()
    
    lip_means = []
    mean_mse_means = []
    std_mse_means = []
    
    for n in n_values:
        sub = df[df['n_train'] == n]
        lip_means.append(sub['lipschitz'].mean())
        # 如果你只跑了旧代码，这两个字段可能没有，需要加上错误处理
        if 'mean_mse' in sub.columns:
            mean_mse_means.append(sub['mean_mse'].mean())
            std_mse_means.append(sub['std_mse'].mean())
        else:
            mean_mse_means.append(0)
            std_mse_means.append(0)
    
    # --- 绘图 (3子图) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.3)

    # 1. Conditional Mean MSE
    ax1.plot(n_values, mean_mse_means, 'o-', color='#05348b', label='Cond. Mean MSE', linewidth=2)
    ax1.set_xlabel(r"Training Set Size $n$", fontsize=16)
    ax1.set_ylabel(r"MSE (Mean)", fontsize=16)
    ax1.set_title("Est. Error of Conditional Mean", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Conditional Std MSE
    ax2.plot(n_values, std_mse_means, 's-', color='#ec813b', label='Cond. Std MSE', linewidth=2)
    ax2.set_xlabel(r"Training Set Size $n$", fontsize=16)
    ax2.set_ylabel(r"MSE (Std)", fontsize=16)
    ax2.set_title("Est. Error of Conditional Std", fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Lipschitz Constant
    log_lip = np.log10(lip_means)
    ax3.plot(n_values, log_lip, 'D-', color='#ae1908', label='Lipschitz Const', linewidth=2)
    ax3.set_xlabel(r"Training Set Size $n$", fontsize=16)
    ax3.set_ylabel(r"Lipschitz constant $log_{10}(L)$", fontsize=16)
    ax3.set_title("Lipschitz Constant vs Sample Size", fontsize=16)
    ax3.set_xscale("log") 
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()

    save_file = os.path.join(result_path, 'cde_full_metrics.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_file}")
    plt.show()

if __name__ == '__main__':
    main()