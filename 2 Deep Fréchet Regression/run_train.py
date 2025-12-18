import multiprocessing
import time
import os
import train_utils 
from tqdm import tqdm  # 引入进度条库

# 强制使用 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 包装函数 (Wrapper)
# ==========================================
# 必须定义在全局层级，以便 spawn 模式下的子进程能 pickle 它
# 作用：将 imap 传入的单个元组参数 (n, m, s) 解包，传给 onedim
def task_wrapper(args):
    return train_utils.onedim(*args)

# ==========================================
# 主执行入口
# ==========================================
if __name__ == '__main__': 
    # 必须显式设置 spawn
    multiprocessing.set_start_method('spawn', force=True) 
    
    # 参数设置
    n_vector = [10, 13, 17, 23, 31, 42, 57, 78, 106, 145, 198, 271, 400]
    m_vector = [20] 
    seeds = list(range(50)) 
    
    # 1. 生成所有计划中的任务
    all_tasks = []
    for n in n_vector:
        for m in m_vector:
            for s in seeds:
                all_tasks.append((n, m, s))
    
    total_tasks_count = len(all_tasks)
    print(f"Total planned tasks: {total_tasks_count}")

    # 2. 断点续跑检查 (Checkpointing Logic)
    # 检查 ./perf/ 目录下是否已有结果，如果有则从任务列表中剔除
    tasks_to_run = []
    skipped_count = 0
    
    os.makedirs("./perf", exist_ok=True)
    
    for (n, m, s) in all_tasks:
        expected_file = f'./perf/n{n}m{m}s{s}.csv'
        if os.path.exists(expected_file):
            skipped_count += 1
        else:
            tasks_to_run.append((n, m, s))
            
    print(f"Found {skipped_count} existing files. Skipping them.")
    print(f"Remaining tasks to run: {len(tasks_to_run)}")
    
    if len(tasks_to_run) == 0:
        print("All tasks completed! Nothing to run.")
        exit(0)

    # 3. 启动并行训练
    # RTX 4090 推荐 nproc = 4，若显存充足可尝试 6 或 8
    nproc = 4
    
    print(f"Starting training on RTX 4090 with {nproc} processes...")
    start_time = time.time()
    
    # 使用 imap_unordered 配合 tqdm 实现进度条
    # 注意：这里我们调用 task_wrapper 而不是直接调用 train_utils.onedim
    with multiprocessing.Pool(processes=nproc) as pool:
        # tqdm 包装迭代器，实现进度显示
        # total 指定总数，ncols 指定宽度，desc 是前缀文字
        results = list(tqdm(
            pool.imap_unordered(task_wrapper, tasks_to_run), 
            total=len(tasks_to_run), 
            desc="Training Progress",
            ncols=100,
            unit="task"
        ))
        
    elapsed = time.time() - start_time
    print(f"\nTraining session completed in {elapsed:.2f} seconds.")
    print(f"Processed {len(tasks_to_run)} tasks.")