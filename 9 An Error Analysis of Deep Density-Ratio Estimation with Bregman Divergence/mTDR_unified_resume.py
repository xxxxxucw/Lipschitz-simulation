# -*- coding: utf-8 -*-
import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from basic_funs_mt_resume import (
    MultivariateNormalDataset,
    accuracy_eval,
    bregmanFNN,
    choose_device,
    compute_lipschitz_constant,
    evaluate_density_ratio_sum,
    load_json,
    lr_bregman,
    save_json,
    setup_seed,
    train_single_stage,
    weight_init,
)


def build_convolution_stage_data(training_q, training_p, validation_q, validation_p, m: int, bridge_num: int):
    a1 = m / bridge_num
    a2 = (m + 1) / bridge_num
    train_q = (np.sqrt(1 - a1 ** 2) * training_q.x + a1 * training_p.x)
    train_p = (np.sqrt(1 - a2 ** 2) * training_q.x + a2 * training_p.x)
    valid_q = (np.sqrt(1 - a1 ** 2) * validation_q.x + a1 * validation_p.x)
    valid_p = (np.sqrt(1 - a2 ** 2) * validation_q.x + a2 * validation_p.x)
    return train_q, train_p, valid_q, valid_p


def build_mixing_delta_arrays(n: int, val_n: int, bridge_num: int, device: torch.device):
    train_delta_array = torch.zeros(n, bridge_num + 1, device=device)
    valid_delta_array = torch.zeros(val_n, bridge_num + 1, device=device)
    for m in range(bridge_num + 1):
        a = m / bridge_num
        sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([a], device=device))
        train_delta_array[:, m] = sampler.sample((n,))[:, 0]
        valid_delta_array[:, m] = sampler.sample((val_n,))[:, 0]
    return train_delta_array, valid_delta_array


def compute_task_errors(model_paths: List[str], dataset_q, width_vec: List[int], device: torch.device):
    pred = evaluate_density_ratio_sum(model_paths, dataset_q.x.to(device), device=device, width_vec=width_vec)
    err = float(accuracy_eval(pred, dataset_q.logpdf.to(device)).detach().cpu().item())
    return err


def run_method(
    method: str,
    n: int,
    seed: int,
    dim: int,
    batch_size: int,
    val_n: int,
    test_n: int,
    lr: float,
    wd: float,
    width_vec: List[int],
    num_epochs: int,
    rho1: float,
    rho2: float,
    bridge_num: int,
    init_num: int,
    patience: int,
    device: torch.device,
    output_root: str,
    verbose: bool,
):
    setup_seed(seed)
    ckpt_dir = os.path.join(output_root, "checkpoints")
    state_dir = os.path.join(output_root, "state")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    state_path = os.path.join(state_dir, f"n{n}_seed{seed}_{method}.json")
    saved_state = load_json(state_path) or {"completed_stages": [], "model_paths": [], "stage_metrics": []}
    completed_stages = set(saved_state.get("completed_stages", []))
    model_paths = list(saved_state.get("model_paths", []))
    stage_metrics = list(saved_state.get("stage_metrics", []))

    training_q = MultivariateNormalDataset(n, dim, rho1, device=device)
    training_p = MultivariateNormalDataset(n, dim, rho2, device=device)
    validation_q = MultivariateNormalDataset(val_n, dim, rho1, device=device)
    validation_p = MultivariateNormalDataset(val_n, dim, rho2, device=device)
    testing_q = MultivariateNormalDataset(test_n, dim, rho1, device=device)
    testing_p = MultivariateNormalDataset(test_n, dim, rho2, device=device)

    if method == "mixing":
        train_delta_array, valid_delta_array = build_mixing_delta_arrays(n, val_n, bridge_num, device)

    stage_iter = tqdm(range(bridge_num), desc=f"{method} | n={n} | seed={seed}", leave=False)
    for m in stage_iter:
        stage_key = f"stage_{m}"
        if stage_key in completed_stages:
            continue

        if method == "convolution":
            train_q_data, train_p_data, valid_q_data, valid_p_data = build_convolution_stage_data(
                training_q, training_p, validation_q, validation_p, m=m, bridge_num=bridge_num
            )
        elif method == "mixing":
            td1 = train_delta_array[:, m].reshape((-1, 1))
            td2 = train_delta_array[:, m + 1].reshape((-1, 1))
            vd1 = valid_delta_array[:, m].reshape((-1, 1))
            vd2 = valid_delta_array[:, m + 1].reshape((-1, 1))
            train_q_data = (1 - td1) * training_q.x + td1 * training_p.x
            train_p_data = (1 - td2) * training_q.x + td2 * training_p.x
            valid_q_data = (1 - vd1) * validation_q.x + vd1 * validation_p.x
            valid_p_data = (1 - vd2) * validation_q.x + vd2 * validation_p.x
        else:
            raise ValueError(f"Unknown method: {method}")

        loader_q = DataLoader(train_q_data, batch_size=batch_size, shuffle=True)
        loader_p = DataLoader(train_p_data, batch_size=batch_size, shuffle=True)

        best_record = None
        best_ckpt = None
        best_score = None
        for init_id in range(init_num):
            net = bregmanFNN(dim=dim, width_vec=width_vec).to(device)
            net.apply(weight_init)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
            ckpt_path = os.path.join(ckpt_dir, f"{method}_n{n}_seed{seed}_stage{m}_init{init_id}.pt")

            record = train_single_stage(
                net=net,
                trainer=optimizer,
                loader_s=loader_p,
                loader_t=loader_q,
                valid_s=valid_p_data,
                valid_t=valid_q_data,
                test_s=testing_p.x,
                test_t=testing_q.x,
                checkpoint_path=ckpt_path,
                num_epochs=num_epochs,
                patience=patience,
                stage_name=f"{method}-n{n}-seed{seed}-stage{m}-init{init_id}",
                verbose=verbose,
            )
            if best_score is None or record.best_valid_loss < best_score:
                best_score = record.best_valid_loss
                best_record = record
                best_ckpt = ckpt_path

        assert best_record is not None and best_ckpt is not None
        model_paths.append(best_ckpt)
        stage_metrics.append(
            {
                "stage": m,
                "best_epoch": best_record.best_epoch,
                "best_train_objective": best_record.best_train_loss,
                "best_valid_objective": best_record.best_valid_loss,
                "best_test_objective": best_record.best_test_loss,
                "best_stage_lipschitz": best_record.best_lipschitz,
                "checkpoint_path": best_ckpt,
            }
        )
        completed_stages.add(stage_key)
        save_json(
            {
                "completed_stages": sorted(list(completed_stages)),
                "model_paths": model_paths,
                "stage_metrics": stage_metrics,
            },
            state_path,
        )

    # final method-level metrics
    train_error = compute_task_errors(model_paths, training_q, width_vec=width_vec, device=device)
    valid_error = compute_task_errors(model_paths, validation_q, width_vec=width_vec, device=device)
    test_error = compute_task_errors(model_paths, testing_q, width_vec=width_vec, device=device)

    # objective values on final estimator
    pred_train = evaluate_density_ratio_sum(model_paths, training_q.x.to(device), device=device, width_vec=width_vec)
    pred_valid = evaluate_density_ratio_sum(model_paths, validation_q.x.to(device), device=device, width_vec=width_vec)
    pred_test = evaluate_density_ratio_sum(model_paths, testing_q.x.to(device), device=device, width_vec=width_vec)
    train_objective = float(lr_bregman(pred_train, torch.zeros_like(pred_train)).detach().cpu().item())
    valid_objective = float(lr_bregman(pred_valid, torch.zeros_like(pred_valid)).detach().cpu().item())
    test_objective = float(lr_bregman(pred_test, torch.zeros_like(pred_test)).detach().cpu().item())

    total_lipschitz = 0.0
    for path in model_paths:
        net = bregmanFNN(dim=dim, width_vec=width_vec).to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        total_lipschitz += compute_lipschitz_constant(net)

    return {
        "n_train": int(n),
        "m_train": 1,
        "seed": int(seed),
        "method": f"{method} bridge",
        "net_id": 0 if method == "convolution" else 1,
        "train_error": float(train_error),
        "valid_error": float(valid_error),
        "test_error": float(test_error),
        "train_loss": float(train_error),
        "valid_loss": float(valid_error),
        "test_loss": float(test_error),
        "train_objective": float(train_objective),
        "valid_objective": float(valid_objective),
        "test_objective": float(test_objective),
        "lipschitz": float(total_lipschitz),
        "num_stages": len(model_paths),
        "bridge_num": int(bridge_num),
    }


def append_result_csv(row: Dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def already_done(csv_path: str, n: int, seed: int, method_label: str) -> bool:
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    mask = (df["n_train"] == n) & (df["seed"] == seed) & (df["method"] == method_label)
    return bool(mask.any())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--output_root", type=str, default="./mtdr_outputs")
    parser.add_argument("--n_list", type=int, nargs="+", default=[500, 1000, 2000, 3000, 5000])
    parser.add_argument("--rep_num", type=int, default=10)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--val_n", type=int, default=5000)
    parser.add_argument("--test_n", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=4000)
    parser.add_argument("--rho1", type=float, default=0.9)
    parser.add_argument("--rho2", type=float, default=0.0)
    parser.add_argument("--bridge_num", type=int, default=5)
    parser.add_argument("--init_num", type=int, default=1)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--seed_base", type=int, default=12412)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"Using device: {device}")

    output_root = args.output_root
    perf_csv = os.path.join(output_root, "perf", "results.csv")
    os.makedirs(os.path.join(output_root, "perf"), exist_ok=True)

    width_vec = [2 * args.dim, 64, 64]
    tasks = []
    for n in args.n_list:
        for rep in range(args.rep_num):
            seed = args.seed_base + rep
            for method in ["convolution", "mixing"]:
                tasks.append((n, seed, method))

    for n, seed, method in tqdm(tasks, desc="Overall progress"):
        method_label = f"{method} bridge"
        if already_done(perf_csv, n, seed, method_label):
            continue
        row = run_method(
            method=method,
            n=n,
            seed=seed,
            dim=args.dim,
            batch_size=args.batch_size,
            val_n=args.val_n,
            test_n=args.test_n,
            lr=args.lr,
            wd=args.wd,
            width_vec=width_vec,
            num_epochs=args.num_epochs,
            rho1=args.rho1,
            rho2=args.rho2,
            bridge_num=args.bridge_num,
            init_num=args.init_num,
            patience=args.patience,
            device=device,
            output_root=output_root,
            verbose=args.verbose,
        )
        append_result_csv(row, perf_csv)

    print(f"Done. Results saved to: {perf_csv}")


if __name__ == "__main__":
    main()
