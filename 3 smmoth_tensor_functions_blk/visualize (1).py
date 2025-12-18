import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import os

# 设置字体，如果系统中没有Times New Roman，会回退到默认字体
try:
    plt.rcParams["font.family"] = "Times New Roman"
except:
    pass
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
    'solid',
    'solid',
    'solid',
    'solid',
    'dashed',
    'dashed'
]

marker = [
    'o',
    's',
    '^',
    'v',
    'D',
    'x'
]

model_name = ['L=2, W=50', 'L=3, W=100', 'L=4, W=200', 'L=5, W=400', 'L=6, W=600', 'L=6, W=800']

# 实验参数设置
n_vector = [10, 13, 17, 23, 31, 42, 57, 78, 106, 145, 198, 271, 400]
fixed_m_train = 20

# 初始化存储矩阵
train_mse_matrix = np.zeros((len(n_vector), 6))  # 存储训练误差
test_mse_matrix = np.zeros((len(n_vector), 6))   # 存储测试误差
lip_matrix = np.zeros((len(n_vector), 6))        # 存储Lipschitz常数

print("开始读取数据...")

# 读取数据循环
for i, n_train in enumerate(n_vector):
    results = []
    for seed in range(50):
        file_path = f"./perf/n{n_train}m{fixed_m_train}s{seed}.csv"
        try:
            if os.path.exists(file_path):
                data = pd.read_csv(file_path, delimiter=',')
                results.append(data)
            # else:
            #     print(f"Warning: File not found {file_path}")
        except Exception as e:
            print(f"Load Data Error: {file_path}, Error: {e}")

    if results:
        # 合并所有种子的结果
        all_results = pd.concat(results, ignore_index=True)
        
        # 按网络ID分组计算均值
        for net_id in range(6):
            net_data = all_results[all_results['net_id'] == net_id]
            if len(net_data) > 0:
                train_mse_matrix[i, net_id] = net_data['train_loss'].mean()
                test_mse_matrix[i, net_id] = net_data['test_loss'].mean()
                lip_matrix[i, net_id] = net_data['lipschitz'].mean()
    else:
        print(f"No results found for n={n_train}, m={fixed_m_train}")

print("数据读取完成，开始绘图...")

# 创建三个子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(top=0.9, bottom=0.15, left=0.08, right=0.95, wspace=0.3)

# 1. 绘制训练误差图 (Linear Scale for Y)
for i in range(6):
    ax1.plot(n_vector, train_mse_matrix[:, i], color=color_tuple[i], linestyle=lines[i],
             label=model_name[i], marker=marker[i], markersize=6)
ax1.set_ylabel(r"$\mathtt{train-mse}$", fontsize=14)
ax1.set_xlabel(r"Training Set Size $n$", fontsize=14)
ax1.set_title(f"Training Error (m={fixed_m_train})", fontsize=16)
ax1.set_xscale("log")  # X轴使用对数坐标，因为n跨度大
ax1.grid(True, which="both", ls="-", alpha=0.3)
# Y轴保持线性，不设置set_yscale

# 2. 绘制测试误差图 (Linear Scale for Y)
for i in range(6):
    ax2.plot(n_vector, test_mse_matrix[:, i], color=color_tuple[i], linestyle=lines[i],
             label=model_name[i], marker=marker[i], markersize=6)
ax2.set_ylabel(r"$\mathtt{test-mse}$", fontsize=14)
ax2.set_xlabel(r"Training Set Size $n$", fontsize=14)
ax2.set_title(f"Test Error (m={fixed_m_train})", fontsize=16)
ax2.set_xscale("log") # X轴使用对数坐标
ax2.grid(True, which="both", ls="-", alpha=0.3)
# Y轴保持线性

# 3. 绘制Lipschitz常数图 (Log Scale for Y: log10(L))
# 处理 log(0) 的情况，加一个极小值 eps
epsilon = 1e-8
log_lip = np.log10(lip_matrix + epsilon)

for i in range(6):
    ax3.plot(n_vector, log_lip[:, i], color=color_tuple[i], linestyle=lines[i],
             label=model_name[i], marker=marker[i], markersize=6)
ax3.set_ylabel(r"Lipschitz constant $\log_{10}(L)$", fontsize=14)
ax3.set_xlabel(r"Training Set Size $n$", fontsize=14)
ax3.set_title(f"Lipschitz Constant (m={fixed_m_train})", fontsize=16)
ax3.set_xscale("log") # X轴使用对数坐标
ax3.grid(True, which="both", ls="-", alpha=0.3)

# 只在最后一个图显示图例，避免遮挡，或者放在图外
# ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.legend(loc='best', fontsize=10) # 也可以选择在第一个图显示

# 保存图片
save_path = './Case1-train-test-20.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
print(f"图片已保存至: {save_path}")

plt.show()