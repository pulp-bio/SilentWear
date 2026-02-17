"""
run_speechnet_ablations_inter_session.py
Runs Inter-Session sweeps over:
- subjects (S01..)
- conditions (silent/vocalized)
- per-block out_channels (number of filters)
- per-block kernel sizes
Skips runs that already have a DONE marker in the deterministic run_tag folder.

"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]   
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(PROJECT_ROOT))
import json

from offline_experiments.inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.general_utils import deepcopy
from offline_experiments.network_ablations.ablations_utils import * 
from models.cnn_architectures.BaseSpeechNet import *

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
        "silent": models_main_dir / "speech_net_base_silent_ablations.csv",
        "vocalized": models_main_dir / "speech_net_base_vocalized_ablations.csv",
    }

    # --------- sweep axes (EDIT THESE)
    subjects = ["S01", "S02", "S03", "S04"]
    #conditions = ["silent", "vocalized"]
    conditions = ["silent", "vocalized"]
    out_channels_sweep = [
        ("C_4-16-16-16-16", [4, 16, 16, 16, 16]),
        ("C_4-8-16-32-64",  [4, 8, 16, 32, 64]),
        ("C_8-8-8-8-8",     [8, 8, 8, 8, 8]),                   # very small
        ("C_8-16-16-32-32", [8, 16, 16, 32, 32]),   
        ("C_16-32-32-64-64", [16, 32, 32, 64, 64]),                
    ]

    kernel_sweep = [
        ("K_default",       [[1, 4], [1, 16], [1, 8], ["full", 1], [1, 1]]),
        ("K_simple",        [[1, 8], [1, 8],  [1, 8], [1, 8],      [1, 8]]),
        ("K_simple_shared", [[2, 8], [2, 8],  [2, 8], [2, 8],      [2, 8]]),            #every 2 channels
    ]

    pool_sweep = [
        ("P_default",        [[1, 8], [1, 4], [1, 4], [1, 1], [1, 1]]),
        ("P_minimal",        [[1, 2], [1, 2], [1, 2], [1, 2], [1, 1]]),
        ("P_minimal_shared", [[2, 2], [2, 2], [1, 2], [1, 2], [1, 1]]),
    ]


    variants = []
    for cname, chans in out_channels_sweep:
        for kname, kernels in kernel_sweep:
            for pname, pools in pool_sweep:
                if not is_combo_valid(kname, pname):
                    #print(f"SKIP (full-kernel would break): {cname}__{kname}__{pname}\n")
                    continue

                blocks_config = build_blocks_config(chans, kernels, pools)
                
                model = BaseSpeechNet(
                    C=14,
                    T=700,
                    output_classes=8,
                    blocks_config=blocks_config,
                    p_dropout=0.1,      # does not matter here
                )
                total, trainable = count_params(model)
                variants.append({
                    "name": f"{cname}_{kname}_{pname}",
                    "channels": chans,
                    "kernels": kernels,
                    "pools": pools,
                    "total_params" : total, 
                    "trainable_params" : trainable, 

                })


    hparams = extract_hparams(deepcopy(base_model_cfg))  # for this ablaiton, hyperparametrs are fixed. could be changed as well
    # ---- run variant-by-variant
    for v in variants:
        variant_name = v["name"]
        variant_total_params = v["total_params"]


        for cond in conditions:
            # run this variant for ALL subjects at this condition
            subject_agg = {}
            for sub in subjects:
                
                cfg_run = deepcopy(base_cfg)
                model_cfg_run = deepcopy(base_model_cfg)

                cfg_run["data"]["subject_id"] = sub
                cfg_run["condition"] = cond

                # deterministic run dir inside subject/condition folder
                cfg_run.setdefault("experiment", {})
                cfg_run["experiment"]["run_tag"] = safe_tag(variant_name)

                # apply ablation to model
                apply_blocks(model_cfg_run, v["channels"], v["kernels"], v["pools"])

                # keep ONE umbrella model name for all ablations 
                model_cfg_run["model"]["name"] = "speechnet_base_abl"

                print("\n" + "=" * 100)
                print(f"SUB={sub} | COND={cond} | VAR={variant_name}")
                print("=" * 100)

                trainer = Inter_Session_Model_Trainer(base_config=cfg_run, model_config=model_cfg_run)
                run_dir = trainer.model_dire

                done_file = run_dir / "DONE"
                running_file = run_dir / "RUNNING"
                failed_file = run_dir / "FAILED"

                # skip completed runs
                if done_file.exists():
                    print("DONE exists → skipping:", run_dir)
                else:
                    # provenance
                    dump_yaml(cfg_run, run_dir / "resolved_base_cfg.yaml")
                    dump_yaml(model_cfg_run, run_dir / "resolved_model_cfg.yaml")
                    write_text(run_dir / "variant.json", json.dumps(v, indent=2))

                    # mark running
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
                        # keep going, but this subject won't contribute to summary
                        continue

                # collect metrics for this subject-condition
                try:
                    stats = summarize_subject_run(run_dir)
                    subject_agg[sub] = stats
                    print(f"   Summary ({stats['metric_col']}): mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n_folds']}")
                except Exception as e:
                    print("    Could not summarize run:", run_dir, "|", repr(e))


                row = {
                    "variant": variant_name,
                    "total_params" : variant_total_params, 
                    "channels_list": json_str(v["channels"]),
                    "kernels_list": json_str(v["kernels"]),
                    "pools_list": json_str(v["pools"]),
                    **hparams,
                }

                # per-subject stats columns
                # You asked: 4 columns one per subject with mean acc, and std across folds
                # We'll store mean + std per subject (so 8 columns total); if you truly want only 4,
                # delete the *_std columns.
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
                row["std_acc_over_subjects"] = float(np.std(means, ddof=1)) if len(means) > 1 else (0.0 if len(means) == 1 else np.nan)

                update_master_csv(master_csv[cond], row, key_cols=("variant",))
                #print(f"\n Updated master CSV for {cond}: {master_csv[cond]}, variant: {master_csv['variant']}")

    print("\nAll ablations finished.")


if __name__ == "__main__":
    main()

