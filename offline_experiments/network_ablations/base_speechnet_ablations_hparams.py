"""
run_speechnet_hparam_ablations_inter_session.py

Runs Inter-Session sweeps over:
- subjects (S01..)
- conditions (silent/vocalized)
- hyperparameters:
    - learning rate (lr)
    - weight decay (weight_decay)
    - dropout (p_dropout)
    - global pooling type (avg / max)

Architecture is FIXED (blocks_config stays unchanged).
Skips runs that already have a DONE marker in the deterministic run_tag folder.

Expected model config structure (example):
model:
  kind: dl
  name: speechnet_base
  kwargs:
    p_dropout: 0.5
    global_pool: avg
    blocks_config: [...]
    train_cfg:
      num_epochs: 50
      optimizer_cfg: {"name": "adam", "lr": 1e-3}
      lr: 1e-3
      weight_decay: 1e-4
"""

import sys
import json
import itertools
from pathlib import Path
import pandas as pd
# --- project root(s)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT2))

import yaml
import numpy as np

from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.general_utils import deepcopy
from offline_experiments.network_ablations.ablations_utils import (
    safe_tag,
    dump_yaml,
    write_text,
    update_master_csv,
    summarize_subject_run,
    extract_hparams,  # optional (we won't rely on it, but keep if you like)
)

# If you want param count once (fixed arch), keep these imports; otherwise remove.
from models.cnn_architectures.BaseSpeechNet import BaseSpeechNet, count_params


def set_nested(dct, keys, value):
    """Safely set dct[keys[0]][keys[1]]...[keys[-1]] = value, creating dicts if missing."""
    cur = dct
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def try_get(dct, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    # --------- config paths (adjust if your script lives elsewhere)
    config_root = Path(__file__).resolve().parent.parent.parent / "config"
    base_config_path = config_root / "base_models_config.yaml"
    model_config_path = config_root / "models_configs" / "speechnet_base.yaml"

    if not base_config_path.exists():
        print("Missing:", base_config_path)
        sys.exit(1)
    if not model_config_path.exists():
        print("Missing:", model_config_path)
        sys.exit(1)

    base_cfg = yaml.safe_load(base_config_path.read_text())
    base_model_cfg = yaml.safe_load(model_config_path.read_text())

    models_main_dir = Path(base_cfg["data"]["models_main_directory"])

    # One master file per condition
    master_csv = {
        "silent": models_main_dir / "speech_net_base_silent_lr_ablations.csv",
        "vocalized": models_main_dir / "speech_net_base_vocalized_lr_ablations.csv",
    }

    # --------- sweep axes (EDIT THESE)
    subjects = ["S01", "S02", "S03", "S04"]
    conditions = ["silent", "vocalized"]

    # Hyperparameter sweep valu

    # sweep values
    lr_sweep = [1e-4, 1e-3]
    wd_sweep = [1e-4]                   # keep yours if already defined
    dropout_sweep = [0.5]       # keep yours if already defined
    global_pool_sweep = ["avg"]    # keep yours if already defined

    scheduler_sweep = ["ReduceLROnPlateau", "none"]
    early_stop_patience_sweep = [10, 5]
    reduce_on_plateau_patience_sweep = [1, 2, 5]


    variants = []
    for lr, wd, dr, gp, sched, es_pat in itertools.product(
        lr_sweep, wd_sweep, dropout_sweep, global_pool_sweep, scheduler_sweep, early_stop_patience_sweep
    ):
        # If scheduler is none, do NOT iterate plateau patience
        rop_pat_list = [None] if sched == "none" else reduce_on_plateau_patience_sweep

        for rop_pat in rop_pat_list:
            # Build variant name (no :g on strings)
            vname = f"lr{lr:g}_wd{wd:g}_do{dr:g}_gp{gp}_es{es_pat}"
            if sched == "none":
                vname += "_schedNone"
            else:
                vname += f"_schedROP_pat{rop_pat}"

            variants.append(
                {
                    "name": vname,
                    "lr": float(lr),
                    "weight_decay": float(wd),
                    "p_dropout": float(dr),
                    "global_pool": str(gp),

                    # scheduler knobs
                    "scheduler": str(sched),  # "none" or "ReduceLROnPlateau"
                    "reduce_on_plateau_patience": (None if rop_pat is None else int(rop_pat)),

                    # early stopping (if your training loop uses it)
                    "early_stop_patience": int(es_pat),
                }
            )
    # --------- build variants (hyperparam combos)
    # lr_sweep = [1e-4, 3e-4, 1e-3]
    # wd_sweep = [0.0, 1e-5, 1e-4]
    # dropout_sweep = [0.0, 0.2, 0.5]
    # global_pool_sweep = ["avg", "max"]  # you said: global vs maxpool (assuming your model uses this flag)
    # variants = []
    # for lr, wd, dr, gp in itertools.product(lr_sweep, wd_sweep, dropout_sweep, global_pool_sweep):
    #     vname = f"lr{lr:g}_wd{wd:g}_do{dr:g}_gp{gp}"
    #     variants.append(
    #         {
    #             "name": vname,
    #             "lr": float(lr),
    #             "weight_decay": float(wd),
    #             "p_dropout": float(dr),
    #             "global_pool": str(gp),
    #         }
    #     )

    # Optional: grab "fixed" hparams from config if you want them written too
    # (Note: extract_hparams() came from your old script; if it doesn't match your new needs,
    # feel free to remove it and just write the sweep params explicitly.)

    # Optional: grab "fixed" hparams from config if you want them written too
    try:
        fixed_hparams = extract_hparams(deepcopy(base_model_cfg))
    except Exception:
        fixed_hparams = {}
    
    for k in ["lr", "weight_decay", "p_dropout", "global_pool", "scheduler",
          "reduce_on_plateau_patience", "early_stop_patience"]:
        fixed_hparams.pop(k, None)

    # ---- run variant-by-variant
    for v0 in variants:
        v = deepcopy(v0)
        variant_name = v["name"]
        lr = float(v["lr"])
        wd = float(v["weight_decay"])
        dr = float(v["p_dropout"])
        gp = str(v["global_pool"])
        sched = str(v["scheduler"])
        rop_pat = v["reduce_on_plateau_patience"]
        es_pat = int(v["early_stop_patience"])


        for cond in conditions:
            subject_agg = {}

            for sub in subjects:
                cfg_run = deepcopy(base_cfg)
                model_cfg_run = deepcopy(base_model_cfg)

                cfg_run["data"]["subject_id"] = sub
                cfg_run["condition"] = cond

                # deterministic run dir inside subject/condition folder
                cfg_run.setdefault("experiment", {})
                cfg_run["experiment"]["run_tag"] = safe_tag(variant_name)

                # keep umbrella model name (optional)
                model_cfg_run["model"]["name"] = "speechnet_base_hparam_abl_lr"

                # --- APPLY HP ABLATION (matches your schema)
                set_nested(model_cfg_run, ["model", "kwargs", "p_dropout"], v["p_dropout"])
                set_nested(model_cfg_run, ["model", "kwargs", "global_pool"], v["global_pool"])

                # train cfg
                set_nested(model_cfg_run, ["model", "kwargs", "train_cfg", "lr"], float(v["lr"]))
                set_nested(model_cfg_run, ["model", "kwargs", "train_cfg", "weight_decay"], float(v["weight_decay"]))
                set_nested(model_cfg_run, ["model", "kwargs", "train_cfg", "optimizer_cfg", "lr"], float(v["lr"]))
                # ---- FORCE early stopping in train_cfg ----
                set_nested(model_cfg_run, ["model", "kwargs", "train_cfg", "early_stop_patience"], int(v["early_stop_patience"]))
                # scheduler: either remove/disable or set config dict
                if v["scheduler"] == "none":
                    # safest: ensure key not present (some trainers treat presence as enabled)
                    try:
                        del model_cfg_run["model"]["kwargs"]["train_cfg"]["scheduler"]
                    except Exception:
                        pass
                else:
                    set_nested(
                        model_cfg_run,
                        ["model", "kwargs", "train_cfg", "scheduler"],
                        {
                            "name": "ReduceLROnPlateau",
                            "mode": "min",
                            "factor": 0.1,
                            "patience": int(v["reduce_on_plateau_patience"]),
                        },
                    )

                print("\n" + "=" * 100)
                print(f"SUB={sub} | COND={cond} | VAR={variant_name}")
                print(
                    f"  lr={v['lr']}, wd={v['weight_decay']}, dropout={v['p_dropout']}, "
                    f"global_pool={v['global_pool']}, scheduler={v['scheduler']}, "
                    f"rop_pat={v['reduce_on_plateau_patience']}, es_pat={v['early_stop_patience']}"
                )
                
                print("=" * 100)

                trainer = Inter_Session_Model_Trainer(base_config=cfg_run, model_config=model_cfg_run)
                run_dir = trainer.model_dire  # keep your original attribute spelling

                done_file = run_dir / "DONE"
                running_file = run_dir / "RUNNING"
                failed_file = run_dir / "FAILED"

                if done_file.exists():
                    print("DONE exists → skipping:", run_dir)
                else:
                    dump_yaml(cfg_run, run_dir / "resolved_base_cfg.yaml")
                    dump_yaml(model_cfg_run, run_dir / "resolved_model_cfg.yaml")
                    write_text(run_dir / "variant.json", json.dumps(v, indent=2))

                    write_text(running_file, "started\n")

                    try:
                        trainer.main()
                        write_text(done_file, "ok\n")
                        if running_file.exists():
                            running_file.unlink()
                        if failed_file.exists():
                            failed_file.unlink()
                        print("Completed:", run_dir)
                    except Exception as e:
                        err = repr(e)
                        write_text(failed_file, err + "\n")
                        if running_file.exists():
                            running_file.unlink()
                        print(" Failed:", run_dir)
                        print("   Error:", err)
                        continue

                # collect metrics for this subject-condition
                try:
                    stats = summarize_subject_run(run_dir)
                    subject_agg[sub] = stats
                    print(
                        f"   Summary ({stats['metric_col']}): mean={stats['mean']:.4f}, "
                        f"std={stats['std']:.4f}, n={stats['n_folds']}"
                    )
                except Exception as e:
                    print("    Could not summarize run:", run_dir, "|", repr(e))

            # ---- write ONE row per (variant, condition) after iterating all subjects

            print("lr writtien to row", lr)
            
            row = {
                **fixed_hparams,
                "variant": variant_name,
                "lr": lr,
                "weight_decay": wd,
                "p_dropout": dr,
                "global_pool": gp,
                "scheduler": sched,
                "reduce_on_plateau_patience": rop_pat,
                "early_stop_patience": es_pat,
                
            }

            for sub in subjects:
                if sub in subject_agg:
                    row[f"{sub}_mean_acc"] = subject_agg[sub]["mean"]
                    row[f"{sub}_std_acc"] = subject_agg[sub]["std"]
                    row[f"{sub}_n_folds"] = subject_agg[sub]["n_folds"]
                    row["metric_col"] = subject_agg[sub]["metric_col"]
                else:
                    row[f"{sub}_mean_acc"] = np.nan
                    row[f"{sub}_std_acc"] = np.nan
                    row[f"{sub}_n_folds"] = np.nan

            means = [subject_agg[s]["mean"] for s in subjects if s in subject_agg]
            row["mean_acc_over_subjects"] = float(np.mean(means)) if means else np.nan
            row["std_acc_over_subjects"] = float(np.std(means, ddof=1)) if len(means) > 1 else (
                0.0 if len(means) == 1 else np.nan
            )

            print("row")
            print(row)
            print("ABOUT TO WRITE row lr =", row["lr"], "variant =", row["variant"])
            update_master_csv(master_csv[cond], row, key_cols=("variant",))
            print(f"Updated master CSV for {cond}: {master_csv[cond]} | variant={variant_name}")
            df_check = pd.read_csv(master_csv[cond])
            print(df_check[df_check["variant"] == variant_name][["variant", "lr"]].tail(1))

if __name__ == "__main__":
    main()
