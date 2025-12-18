import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import os

# 设置字体和样式
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=15)
rc('text', usetex=False)

# 定义颜色和线型
color_tuple = [
    '#ae1908',  # red
    '#ec813b',  # orange
    '#05348b',  # dark blue
    '#9acdc4',  # pain blue
    '#00FF00',  # green
    '#0000FF'   # blue
]

lines = [
    'solid', 'solid', 'solid', 'solid', 'dashed', 'dashed'
]

marker = [
    'o', 's', '^', 'v', 'D', 'x'
]

# 对应 train_ts.py 中的 net_configs
model_name = [
    'L=2, W=50',
    'L=3, W=100', 
    'L=4, W=200', 
    'L=5, W=400', 
    'L=6, W=600', 
    'L=6, W=800'
]

# 实验参数设置 (必须与 train_ts.py 一致)
n_vector = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
fixed_m_train = 1
num_seeds = 50

# 初始化存储矩阵 [n_sizes, n_models]
train_mse_matrix = np.zeros((len(n_vector), 6))
test_mse_matrix = np.zeros((len(n_vector), 6))
lip_matrix = np.zeros((len(n_vector), 6))

print("Loading data...")

for i, n_train in enumerate(n_vector):
    results = []
    for seed in range(num_seeds):
        file_path = f"./perf/n{n_train}m{fixed_m_train}s{seed}.csv"
        if os.path.exists(file_path):
            try:
                # 读取CSV，确保文件不为空
                if os.path.getsize(file_path) > 0:
                    data = pd.read_csv(file_path)
                    results.append(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
    if results:
        # 合并当前 n 下所有 seed 的结果
        all_results = pd.concat(results, ignore_index=True)
        
        # 按 net_id (0-5) 分组计算均值
        for net_id in range(6):
            net_data = all_results[all_results['net_id'] == net_id]
            if len(net_data) > 0:
                train_mse_matrix[i, net_id] = net_data['train_loss'].mean()
                test_mse_matrix[i, net_id] = net_data['test_loss'].mean()
                lip_matrix[i, net_id] = net_data['lipschitz'].mean()
            else:
                # 如果某个模型完全没有数据，填充NaN避免画图错误连线
                train_mse_matrix[i, net_id] = np.nan
                test_mse_matrix[i, net_id] = np.nan
                lip_matrix[i, net_id] = np.nan
    else:
        print(f"Warning: No valid data found for n={n_train}")
        train_mse_matrix[i, :] = np.nan
        test_mse_matrix[i, :] = np.nan
        lip_matrix[i, :] = np.nan

print("Data loaded. Plotting...")

# 创建三个子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.95, wspace=0.3)

# --- 1. 绘制训练误差 (Linear Scale) ---
for i in range(6):
    # 过滤掉NaN值进行绘图
    mask = ~np.isnan(train_mse_matrix[:, i])
    if mask.any():
        ax1.plot(np.array(n_vector)[mask], train_mse_matrix[mask, i], 
                 color=color_tuple[i], linestyle=lines[i],
                 label=model_name[i], marker=marker[i], markersize=6)

ax1.set_ylabel(r"Training MSE", fontsize=16)
ax1.set_xlabel(r"Training Set Size $n$", fontsize=16)
ax1.set_title(f"Training Error", fontsize=18)
ax1.set_xscale("log") # X轴对数刻度以便观察大范围n
ax1.grid(True, which="both", ls="-", alpha=0.2)
# ax1.legend() # 图例太多可以只在最后一个图显示

# --- 2. 绘制测试误差 (Linear Scale) ---
for i in range(6):
    mask = ~np.isnan(test_mse_matrix[:, i])
    if mask.any():
        ax2.plot(np.array(n_vector)[mask], test_mse_matrix[mask, i], 
                 color=color_tuple[i], linestyle=lines[i],
                 label=model_name[i], marker=marker[i], markersize=6)

ax2.set_ylabel(r"Test MSE", fontsize=16)
ax2.set_xlabel(r"Training Set Size $n$", fontsize=16)
ax2.set_title(f"Test Error", fontsize=18)
ax2.set_xscale("log")
ax2.grid(True, which="both", ls="-", alpha=0.2)


# --- 3. 绘制 Lipschitz 常数 (Log Scale) ---
# 计算 Log10，处理可能的 0 或 NaN
valid_lip_mask = lip_matrix > 0
log_lip = np.zeros_like(lip_matrix)
log_lip[:] = np.nan
log_lip[valid_lip_mask] = np.log10(lip_matrix[valid_lip_mask])

for i in range(6):
    mask = ~np.isnan(log_lip[:, i])
    if mask.any():
        ax3.plot(np.array(n_vector)[mask], log_lip[mask, i], 
                 color=color_tuple[i], linestyle=lines[i],
                 label=model_name[i], marker=marker[i], markersize=6)

ax3.set_ylabel(r"Lipschitz Constant $\log_{10}(L)$", fontsize=16)
ax3.set_xlabel(r"Training Set Size $n$", fontsize=16)
ax3.set_title(f"Lipschitz Constant", fontsize=18)
ax3.set_xscale("log")
ax3.grid(True, which="both", ls="-", alpha=0.2)
ax3.legend(fontsize=12, loc='upper left')

# 保存图片
save_dir = './Simulation/Case1'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'Case1-TS-Result.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')

print(f"Figure saved to {save_path}")
plt.show()