#!/usr/bin/env python3
"""
III_ft_results.py

Generate results for incremental fine-tuning (FT) vs training-from-scratch (baseline) models.

This version is a drop-in replacement of your script, but:
- Adds argparse so it can be run from CLI (no manual "USER EDITABLE PART").
- Adds deterministic model_name_id = f"w{window_ms}ms" derived from run_cfg window size,
  matching the logic used in the other scripts.
- Fixes two subtle issues:
  1) summary_subject dict was being re-created inside the group loop (now created after loop)
  2) baseline std across subjects was computed after appending the mean row (now computed before)

Folder assumptions (unchanged from your script):
- FT results:
  <ARTIFACTS_DIR>/<Sxx>/<condition>/<model_name>/<model_base_id>/<ft_id>/ft_summary.csv
  plus ft_cfg.json and run_cfg.json at:
  <ARTIFACTS_DIR>/<Sxx>/<condition>/<model_name>/<model_base_id>/<ft_id>/ft_cfg.json
  <ARTIFACTS_DIR>/<Sxx>/<condition>/<model_name>/<model_base_id>/run_cfg.json

- Baseline results:
  <ARTIFACTS_DIR>/baseline_models/<Sxx>/<condition>/<model_name>/<model_base_id>/<bs_id>/train_from_scratch_summary.csv
  plus tfs_cfg.json and run_cfg_min.json at:
  <ARTIFACTS_DIR>/baseline_models/<Sxx>/<condition>/<model_name>/<model_base_id>/<bs_id>/tfs_cfg.json
  <ARTIFACTS_DIR>/baseline_models/<Sxx>/<condition>/<model_name>/<model_base_id>/<bs_id>/run_cfg_min.json

Outputs:
- Tables:
  <ARTIFACTS_DIR>/tables/ft_bs_results_<condition>_<model_name>_<win_ms>.csv
- Figures:
  <ARTIFACTS_DIR>/figures/ft_<ft_id>_<condition>.png
  <ARTIFACTS_DIR>/figures/avg_<condition>_<model_name>_w<win_ms>ms.pdf
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches

# Project-level imports
project_root = Path().resolve()
sys.path.insert(0, str(project_root))
from utils.general_utils import open_file


# ------------------------- small helpers -------------------------

def _model_name_id_from_window_ms(win_ms: int) -> str:
    return f"w{int(win_ms)}ms"


def _fmt_sci(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if (ax < 1e-2) or (ax >= 1e3):
        return f"{x:.0e}"
    return f"{x:g}"


def _cond_title(cond: str) -> str:
    c = cond.lower()
    if c == "silent":
        return "Silent"
    if c == "vocalized":
        return "Vocalized"
    return cond[:1].upper() + cond[1:]


def _ensure_sorted_by_x(summary_subject: dict) -> dict:
    """Sort arrays by num_prev_ft_rounds to guarantee consistent plotting/tabulation."""
    x = np.asarray(summary_subject["num_prev_ft_rounds"])
    order = np.argsort(x)
    out = dict(summary_subject)
    out["num_prev_ft_rounds"] = x[order]
    out["subj_acc_means"] = np.asarray(summary_subject["subj_acc_means"])[order]
    out["subjs_acc_std"] = np.asarray(summary_subject["subjs_acc_std"])[order]
    out["subjs_acc_means_noft"] = np.asarray(summary_subject["subjs_acc_means_noft"])[order]
    out["subjs_acc_std_noft"] = np.asarray(summary_subject["subjs_acc_std_noft"])[order]
    return out


# ------------------------- core loaders -------------------------

def load_results(
    base_model_folder,
    model_name,
    condition,
    subjects,
    ft_id,
    model_base_id,
    inter_session_id = None, 
    type="ft",
    ):
    """
    Loads per-subject arrays over num_prev_ft_rounds, both for fine tuning and baseline experiments.
    Enforces that lr/scheduler/num_epochs are identical across subjects for a given ft_id.

    Returns: list of per-subject dicts.
    """
    summary_condition_across_subjects = []

    ref_lr = None
    ref_sched = None
    ref_epochs = None

    for subject_id in subjects:

        if type == "train_from_scratch":
            results_path = (Path(base_model_folder)/ f"{subject_id}/{condition}/{model_name}/{model_base_id}/{ft_id}/{type}_summary.csv")
            cfg_file = (
                Path(base_model_folder)
                / f"{subject_id}/{condition}/{model_name}/{model_base_id}/{ft_id}/tfs_cfg.json"
            )
            run_cfg_file = (
                Path(base_model_folder)
                / f"{subject_id}/{condition}/{model_name}/{model_base_id}/{ft_id}/run_cfg_min.json"
            )
            if not run_cfg_file.exists():
                raise FileNotFoundError(f"Missing run_cfg_min.json: {run_cfg_file}")
            min_run_cfg = open_file(run_cfg_file)
            win_size = int(float(min_run_cfg["base_cfg"]["window"]["window_size_s"]) * 1000)

        else:
            results_path = (Path(base_model_folder)/ f"{subject_id}/{condition}/{model_name}/{model_base_id}/{ft_id}/{type}_summary.csv")
            cfg_file = (
                Path(base_model_folder)
                / f"{subject_id}/{condition}/{model_name}/{model_base_id}/{ft_id}/ft_cfg.json"
            )
            run_cfg_file = (
                Path(base_model_folder)
                / f"{subject_id}/{condition}/{model_name}/{model_base_id}/{inter_session_id}/run_cfg.json"
            )
            if not run_cfg_file.exists():
                raise FileNotFoundError(f"Missing run_cfg.json: {run_cfg_file}")
            min_run_cfg = open_file(run_cfg_file)
            win_size = int(float(min_run_cfg["base_cfg"]["window"]["window_size_s"]) * 1000)


        model_name_id = _model_name_id_from_window_ms(win_size)

        if not results_path.exists():
            raise FileNotFoundError(f"Missing {type}_summary.csv: {results_path}")
        if not cfg_file.exists():
            raise FileNotFoundError(f"Missing cfg json: {cfg_file}")

        cfg = open_file(cfg_file)
        train_cfg = cfg["model"]["kwargs"]["train_cfg"]

        # scheduler name
        sched_cfg = train_cfg.get("scheduler", None)
        sched_name = (
            sched_cfg["name"]
            if isinstance(sched_cfg, dict) and "name" in sched_cfg
            else "no_sched"
        )

        lr = float(train_cfg["lr"])
        num_epochs = int(train_cfg["num_epochs"])

        # enforce hyperparam consistency across subjects for this ft_id
        if ref_lr is None:
            ref_lr, ref_sched, ref_epochs = lr, sched_name, num_epochs
        else:
            mismatches = []
            if lr != ref_lr:
                mismatches.append(f"lr={lr} (ref {ref_lr})")
            if sched_name != ref_sched:
                mismatches.append(f"scheduler={sched_name} (ref {ref_sched})")
            if num_epochs != ref_epochs:
                mismatches.append(f"num_epochs={num_epochs} (ref {ref_epochs})")
            if mismatches:
                raise ValueError(
                    f"Hyperparams mismatch for id={ft_id}, condition={condition}.\n"
                    f"Subject {subject_id} differs: " + ", ".join(mismatches) + "\n"
                    f"Config file: {cfg_file}"
                )

        summary_csv = pd.read_csv(results_path)

        # groups define folds/repeats: group over (num_prev_ft_rounds, zero_shot_test_batch)
        groups = summary_csv.groupby(["num_prev_ft_rounds", "zero_shot_test_batch"])

        num_prev_ft_rounds = []
        acc_means = []
        acc_stds = []
        acc_noft_means = []
        acc_noft_stds = []

        for _, group_df in groups:
            num_prev_ft_rounds.append(int(group_df["num_prev_ft_rounds"].iloc[0]))

            zs = group_df["zero_shot_balanced_acc"].values
            acc_means.append(np.mean(zs))
            acc_stds.append(np.std(zs))

            if type == "ft":
                noft = group_df["balanced_acc_no_ft"].values
                acc_noft_means.append(np.mean(noft))
                acc_noft_stds.append(np.std(noft))

        # IMPORTANT: build dict AFTER loop
        n = len(num_prev_ft_rounds)
        summary_subject = {
            "subject_id": subject_id,
            "subj_acc_means": np.array(acc_means) * 100.0,
            "subjs_acc_std": np.array(acc_stds) * 100.0,
            "subjs_acc_means_noft": (
                np.array(acc_noft_means) * 100.0 if type == "ft" else np.full(n, np.nan)
            ),
            "subjs_acc_std_noft": (
                np.array(acc_noft_stds) * 100.0 if type == "ft" else np.full(n, np.nan)
            ),
            "num_prev_ft_rounds": np.array(num_prev_ft_rounds),
            "lr": lr,
            "scheduler": sched_name,
            "num_ft_epochs": num_epochs,
            "win_size_ms": win_size,
            "model_name_id": model_name_id,
        }

        summary_subject = _ensure_sorted_by_x(summary_subject)
        summary_condition_across_subjects.append(summary_subject)

    return summary_condition_across_subjects


# ------------------------- tabulation -------------------------

def summarize_subject_table(summary_condition_across_subjects, ft_id, condition):
    """
    Builds a table with per-subject mean/std for each prev_ft_round, for:
      - Fine Tuning (FT)
      - Zero Shot on intersession model (NOFT baseline)
    Also includes global averages across subjects.
    """
    x = summary_condition_across_subjects[0]["num_prev_ft_rounds"].tolist()

    rows = []
    for subj in summary_condition_across_subjects:
        for i, r in enumerate(x):
            rows.append(
                {
                    "condition": condition,
                    "ft_id": ft_id,
                    "subject": subj["subject_id"],
                    "prev_ft_rounds": int(r),
                    "FT_mean": float(subj["subj_acc_means"][i]),
                    "FT_std": float(subj["subjs_acc_std"][i]),
                    "NOFT_mean": float(subj["subjs_acc_means_noft"][i]),
                    "NOFT_std": float(subj["subjs_acc_std_noft"][i]),
                }
            )

    df = pd.DataFrame(rows)

    def _global_stats(group):
        ft_means = group["FT_mean"].values
        zs_means = group["NOFT_mean"].values
        return pd.Series(
            {
                "FT_mean_global": float(np.mean(ft_means)),
                "FT_std_global": float(np.std(ft_means)),
                "NOFT_mean_global": float(np.mean(zs_means)),
                "NOFT_std_global": float(np.std(zs_means)),
            }
        )

    df_global = df.groupby(
        ["condition", "ft_id", "prev_ft_rounds"], as_index=False
    )[["FT_mean", "NOFT_mean"]].apply(_global_stats)

    return df, df_global


def summary_to_csv(summary_ft, summary_baseline, res_save_folder, condition, model_to_select):
    """
    Writes a wide CSV containing per-subject FT and baseline arrays + an Average row.
    """
    df_summary = pd.DataFrame(summary_ft)

    # ---- FT mean across subjects (by column) ----
    subjs_accs_means = np.vstack(df_summary["subj_acc_means"].values)
    mean_across_subjs = np.mean(subjs_accs_means, axis=0)

    subjs_accs_means_noft = np.vstack(df_summary["subjs_acc_means_noft"].values)
    mean_across_subjs_no_ft = np.mean(subjs_accs_means_noft, axis=0)

    new_row = {
        "subject_id": "Average",
        "subj_acc_means": [mean_across_subjs],
        "subjs_acc_means_noft": [mean_across_subjs_no_ft],
        "subjs_acc_std": [np.std(subjs_accs_means, axis=0)],
        "subjs_acc_std_noft": [np.std(subjs_accs_means_noft, axis=0)],
        "num_prev_ft_rounds": [df_summary["num_prev_ft_rounds"].iloc[0]],
        "lr": [df_summary["lr"].iloc[0]],
        "scheduler": [df_summary["scheduler"].iloc[0]],
        "num_ft_epochs": [df_summary["num_ft_epochs"].iloc[0]],
        "win_size_ms": [df_summary["win_size_ms"].iloc[0]],
        "model_name_id": [df_summary["model_name_id"].iloc[0]],
    }
    df_summary = pd.concat((df_summary, pd.DataFrame(new_row)), ignore_index=True)

    # ---- Baseline arrays ----
    df_summary_baseline = pd.DataFrame(summary_baseline)

    per_subject_means_baseline = np.vstack(df_summary_baseline["subj_acc_means"].values)
    per_subject_stds_baseline = np.vstack(df_summary_baseline["subjs_acc_std"].values)

    # IMPORTANT FIX: compute std across subjects BEFORE appending the mean row
    std_across_subjs_baseline = per_subject_means_baseline.std(axis=0)
    mean_across_subjs_baseline = per_subject_means_baseline.mean(axis=0)

    per_subject_means_baseline = np.vstack([per_subject_means_baseline, mean_across_subjs_baseline])
    per_subject_stds_baseline = np.vstack([per_subject_stds_baseline, std_across_subjs_baseline])

    df_summary["subj_acc_means_fromscratch"] = list(per_subject_means_baseline)
    df_summary["subj_acc_std_fromscratch"] = list(per_subject_stds_baseline)

    unique_win = df_summary["win_size_ms"].dropna().unique()
    if len(unique_win) != 1:
        print("ERROR: win size not unique", unique_win)
        sys.exit(1)

    win_size = int(unique_win[0])
    res_save_folder = Path(res_save_folder)
    res_save_folder.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(res_save_folder / f"ft_bs_results_{condition}_{model_to_select}_{win_size}.csv", index=False)
    return df_summary


# ------------------------- alignment helpers for plotting -------------------------

def _idx_for(s, x_target):
    sx = np.asarray(s["num_prev_ft_rounds"])
    return np.array([np.where(sx == v)[0][0] for v in x_target])


def _align_common_x(ft_by, sc_by, subject_ids):
    per_subj_common = []
    for sid in subject_ids:
        x_ft = np.asarray(ft_by[sid]["num_prev_ft_rounds"])
        x_sc = np.asarray(sc_by[sid]["num_prev_ft_rounds"])
        per_subj_common.append(np.intersect1d(x_ft, x_sc))

    common_x = sorted(set(per_subj_common[0]).intersection(*map(set, per_subj_common[1:])))
    common_x = np.asarray(common_x)

    if common_x.size == 0:
        raise ValueError("No common num_prev_ft_rounds across subjects between FT and Scratch.")

    return common_x


def prepare_aligned(ft_summary, scratch_summary, show_no_ft=True):
    """
    Align FT and Scratch summaries on a common x and return:
      - x
      - subject_ids
      - per_subj dict with aligned arrays
      - info_ft, info_scratch
      - avg curves (mean/std across subjects) for ft/noft/scratch
    """
    info_ft = {
        "lr": ft_summary[0]["lr"],
        "num_epochs": ft_summary[0]["num_ft_epochs"],
        "scheduler": ft_summary[0]["scheduler"],
    }
    info_sc = {
        "lr": scratch_summary[0]["lr"],
        "num_epochs": scratch_summary[0]["num_ft_epochs"],
        "scheduler": scratch_summary[0]["scheduler"],
    }

    ft_by = {s["subject_id"]: s for s in ft_summary}
    sc_by = {s["subject_id"]: s for s in scratch_summary}
    subject_ids = sorted(set(ft_by.keys()) & set(sc_by.keys()))
    if not subject_ids:
        raise ValueError("No overlapping subject_ids between FT and Scratch summaries.")

    x = _align_common_x(ft_by, sc_by, subject_ids)

    per_subj = {}
    for sid in subject_ids:
        sft = ft_by[sid]
        ssc = sc_by[sid]
        i_ft = _idx_for(sft, x)
        i_sc = _idx_for(ssc, x)

        per_subj[sid] = dict(
            x=x,
            ft_mean=np.asarray(sft["subj_acc_means"])[i_ft],
            ft_std=np.asarray(sft["subjs_acc_std"])[i_ft],
            nf_mean=np.asarray(sft["subjs_acc_means_noft"])[i_ft] if show_no_ft else None,
            nf_std=np.asarray(sft["subjs_acc_std_noft"])[i_ft] if show_no_ft else None,
            sc_mean=np.asarray(ssc["subj_acc_means"])[i_sc],
            sc_std=np.asarray(ssc["subjs_acc_std"])[i_sc],
        )

    ft_stack = np.stack([per_subj[sid]["ft_mean"] for sid in subject_ids], axis=0)
    sc_stack = np.stack([per_subj[sid]["sc_mean"] for sid in subject_ids], axis=0)

    ft_mean = ft_stack.mean(axis=0)
    ft_std = np.std(ft_stack, axis=0)
    sc_mean = sc_stack.mean(axis=0)
    sc_std = np.std(sc_stack, axis=0)

    nf_mean = nf_std = None
    if show_no_ft:
        nf_stack = np.stack([per_subj[sid]["nf_mean"] for sid in subject_ids], axis=0)
        nf_mean = nf_stack.mean(axis=0)
        nf_std = np.std(nf_stack, axis=0)

    avg = dict(
        ft_mean=ft_mean,
        ft_std=ft_std,
        nf_mean=nf_mean,
        nf_std=nf_std,
        sc_mean=sc_mean,
        sc_std=sc_std,
    )
    return x, subject_ids, per_subj, info_ft, info_sc, avg


# ------------------------- plotting -------------------------


def _slice_to_x(subj_dict, x_target):
    """Return arrays aligned to x_target by indexing."""
    x = np.asarray(subj_dict["num_prev_ft_rounds"])
    idx = np.array([np.where(x == v)[0][0] for v in x_target])
    return (
        x_target,
        np.asarray(subj_dict["subj_acc_means"])[idx],
        np.asarray(subj_dict["subjs_acc_std"])[idx],
        np.asarray(subj_dict["subjs_acc_means_noft"])[idx] if "subjs_acc_means_noft" in subj_dict else None,
        np.asarray(subj_dict["subjs_acc_std_noft"])[idx] if "subjs_acc_std_noft" in subj_dict else None,
    )


def plot_subjs_and_avgs(ft_summary, scratch_summary, show_no_ft=True, save_path=None):
    """
    1 x (Nsubjects+1) layout: each subject is one panel + an "Average" panel at the end.
    Keeps your outer-box styling and hides per-axes spines.
    """
    x, subject_ids, per_subj, info_ft, info_sc, avg = prepare_aligned(ft_summary, scratch_summary, show_no_ft)

    n_subj = len(subject_ids)
    ncols = n_subj + 1
    fig, axes = plt.subplots(1, ncols, figsize=(40, 8), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    fs_ax, fs_label, fs_tick, fs_leg = 30, 30, 30, 30
    color_ft = "blue"
    color_intersess = "red"
    color_scratch = "green"

    # --- per-subject plots ---
    for i, (ax, sid) in enumerate(zip(axes[:n_subj], subject_ids)):
        face = "#f4f4f4" if (i % 2 == 1) else "#ffffff"
        ax.set_facecolor(face)

        d = per_subj[sid]

        ax.errorbar(d["x"] + 1, d["ft_mean"], yerr=d["ft_std"],
                    marker="o", linestyle="-", capsize=4, linewidth=2, color=color_ft)
        if show_no_ft:
            ax.errorbar(d["x"] + 1, d["nf_mean"], yerr=d["nf_std"],
                        marker="o", linestyle="-", capsize=4, linewidth=2, color=color_intersess)

        # From scratch: don't show model 0 (random guessing)
        x_sc = np.asarray(d["x"] + 1)
        ax.errorbar(x_sc[1:], np.asarray(d["sc_mean"])[1:], np.asarray(d["sc_std"])[1:],
                    marker="s", linestyle="-", capsize=4, linewidth=2, color=color_scratch)

        ax.set_title(sid, fontsize=fs_ax, y=0.92)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xticks(x + 1)
        ax.tick_params(axis="both", labelsize=fs_tick)

    # --- average panel ---
    ax_avg = axes[-1]
    face = "#f4f4f4" if (n_subj % 2 == 1) else "#ffffff"
    ax_avg.set_facecolor(face)

    ax_avg.errorbar(x + 1, avg["ft_mean"], yerr=avg["ft_std"],
                    marker="o", linestyle="-", capsize=4, linewidth=1.5, color=color_ft)

    if show_no_ft:
        ax_avg.errorbar(x + 1, avg["nf_mean"], yerr=avg["nf_std"],
                        marker="o", linestyle="-", capsize=4, linewidth=1.5, color=color_intersess)

    x_sc = x + 1
    ax_avg.errorbar(x_sc[1:], avg["sc_mean"][1:], avg["sc_std"][1:],
                    marker="s", linestyle="-", capsize=4, linewidth=1.5, color=color_scratch)

    ax_avg.set_title("Average", fontsize=fs_ax, y=0.92)
    ax_avg.grid(True, linestyle="--", alpha=0.4)
    ax_avg.set_ylim(0, 100)
    ax_avg.set_yticks(np.arange(0, 101, 10))
    ax_avg.set_xticks(x + 1)
    ax_avg.tick_params(axis="both", labelsize=fs_tick)

    # --- legend ---
    style_handles = [Line2D([0], [0], color=color_ft, lw=2.6, linestyle="-", marker="o", label="Fine Tuning")]
    if show_no_ft:
        style_handles.append(Line2D([0], [0], color=color_intersess, lw=2.6, linestyle="-", marker="o", label="No Fine Tuning"))
    style_handles.append(Line2D([0], [0], color=color_scratch, lw=2.6, linestyle="-", marker="s", label="Train From Scratch"))

    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(style_handles),
        frameon=False,
        fontsize=fs_leg,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=0.8,
        labelspacing=0.2,
        borderaxespad=0.0,
    )

    # --- layout ---
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.13, top=0.86, wspace=0.05)
    fig.supylabel("Accuracy (%)", fontsize=fs_label, x=0.04)
    fig.supxlabel("Last tested batch", fontsize=fs_label, y=0.03)

    fig.canvas.draw()

    # hide per-axes spines and keep y-ticks only on left edge
    x0s = np.array([ax.get_position().x0 for ax in axes])
    left_x0 = x0s.min()
    for ax in axes:
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")

        is_left_edge = np.isclose(ax.get_position().x0, left_x0, atol=1e-3)
        if is_left_edge:
            ax.tick_params(axis="y", left=True, labelleft=True, pad=6)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    # outer box around plotting area
    x0 = min(ax.get_position().x0 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)

    outer_box = patches.Rectangle(
        (x0 - 0.001, y0 - 0.002),
        (x1 - x0) + 0.001 + 0.002,
        (y1 - y0) + 0.002 + 0.002,
        transform=fig.transFigure,
        fill=False,
        linewidth=0.2,
        edgecolor="black",
        zorder=10,
    )
    fig.add_artist(outer_box)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return info_ft, info_sc


# ------------------------- CLI main -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="Root artifacts dir (default: env SILENTWEAR_ARTIFACTS_DIR or ./artifacts)",
    )

    parser.add_argument("--model_name", type=str, default="speechnet")
    parser.add_argument("--model_base_id", type=str, default="w1400ms")
    parser.add_argument("--inter_session_model_id", type=str, default="model_1")
    parser.add_argument("--ft_id", type=str, default="ft_config_0")
    parser.add_argument("--bs_id", type=str, default="bs_config_0")

    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    parser.add_argument("--conditions", nargs="+", default=["vocalized", "silent"])

    args = parser.parse_args()

    # Resolve ARTIFACTS_DIR
    if args.artifacts_dir is not None:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = Path(os.environ.get("SILENTWEAR_ARTIFACTS_DIR", project_root / "artifacts"))

    base_ft_model_folder = artifacts_dir/"models"/"inter_session_ft"
    base_from_scratch_model_folder = artifacts_dir/"models"/"train_from_scratch"
    res_save_folder = artifacts_dir / "tables"
    fig_save_folder = artifacts_dir / "figures"
    res_save_folder.mkdir(parents=True, exist_ok=True)
    fig_save_folder.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        # ---- FT ----
        summary_ft = load_results(
            base_model_folder=base_ft_model_folder,
            model_name=args.model_name,
            condition=condition,
            subjects=args.subjects,
            ft_id=args.ft_id,
            model_base_id=args.model_base_id,
            inter_session_id = args.inter_session_model_id,
            type="ft",
        )
        
        # ---- Baseline ----
        summary_baseline = load_results(
            base_model_folder=base_from_scratch_model_folder,
            model_name=args.model_name,
            condition=condition,
            subjects=args.subjects,
            ft_id=args.bs_id,
            model_base_id=args.model_base_id,
            inter_session_id = None, 
            type="train_from_scratch",
        )

        df_summary = summary_to_csv(summary_ft, summary_baseline, res_save_folder, condition, args.model_name)

        win_size = int(df_summary["win_size_ms"].dropna().unique()[0])
        model_name_id = _model_name_id_from_window_ms(win_size)

        plot_subjs_and_avgs(
            ft_summary=summary_ft,
            scratch_summary=summary_baseline,
            show_no_ft=True,
            save_path=fig_save_folder / f"avg_{condition}_{args.model_name}_{model_name_id}.svg",
        )


if __name__ == "__main__":
    main()
