"""
Script to analyze fine-tuning results + compare multiple FT configs.

- Plots ONLY the average across subjects for multiple ft_config_ids (same condition).
- Prints a table with per-subject values (mean/std across folds) + global averages.
- Returns lr, num_epochs (and scheduler) per ft_config_id.
"""
import os
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# notebook location
project_root = Path().resolve()
ARTIFACTS_DIR = Path(os.environ.get('SILENTWEAR_ARTIFACTS_DIR', project_root / 'artifacts'))
.parent.parent
sys.path.insert(0, str(project_root))
from utils.general_utils import open_file


##################### EDIT HERE ####################################################
fig_path = ARTIFACTS_DIR

base_ft_model_folder = ARTIFACTS_DIR

model_base_id = "model_1"
ft_config_ids = ["ft_config_0"]                 # seems to be the best

baseline_model_folder =  ARTIFACTS_DIR
baseline_configs_ids = ["bs_config_0"]

subjects = ["S01", "S02", "S03", "S04"]
conditions = ["vocalized", "silent"]
####################################################################################


def _fmt_sci(x: float) -> str:
    """Format 0.001 -> 1e-3, 1.0 -> 1, etc."""
    if x is None:
        return "NA"
    if x == 0:
        return "0"
    ax = abs(x)
    if (ax < 1e-2) or (ax >= 1e3):
        return f"{x:.0e}"
    return f"{x:g}"


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


def load_results(base_model_folder, condition, subjects, ft_id, model_base_id, type='ft'):
    """
    Loads per-subject arrays over num_prev_ft_rounds.
    Enforces that lr/scheduler/num_epochs are identical across subjects for a given ft_id.
    """
    summary_condition_across_subjects = []

    ref_lr = None
    ref_sched = None
    ref_epochs = None

    for subject_id in subjects:

        results_path = (Path(base_model_folder)/ f"{subject_id}/{condition}/speechnet_base/{model_base_id}/{ft_id}/{type}_summary.csv")
        if type=='baseline':
            ft_cfg_file = (Path(base_model_folder)/ f"{subject_id}/{condition}/speechnet_base/{model_base_id}/{ft_id}/bs_cfg.json")
        else:
            ft_cfg_file = (Path(base_model_folder)/ f"{subject_id}/{condition}/speechnet_base/{model_base_id}/{ft_id}/ft_cfg.json")



        if not results_path.exists():
            raise FileNotFoundError(f"Missing ft_summary.csv: {results_path}")
        if not ft_cfg_file.exists():
            raise FileNotFoundError(f"Missing ft_cfg.json: {ft_cfg_file}")

        ft_cfg = open_file(ft_cfg_file)
        train_cfg = ft_cfg["model"]["kwargs"]["train_cfg"]

        # scheduler name
        sched_cfg = train_cfg.get("scheduler", None)
        sched_name = (
            sched_cfg["name"]
            if isinstance(sched_cfg, dict) and "name" in sched_cfg
            else "no_sched"
        )

        lr = float(train_cfg["lr"])
        num_epochs = int(train_cfg["num_epochs"])

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
                    f"FT hyperparams mismatch for ft_id={ft_id}, condition={condition}.\n"
                    f"Subject {subject_id} differs: " + ", ".join(mismatches) + "\n"
                    f"Config file: {ft_cfg_file}"
                )

        summary_ft_csv = pd.read_csv(results_path)

        # groups define folds/repeats: group over (num_prev_ft_rounds, zero_shot_test_batch)
        groups = summary_ft_csv.groupby(["num_prev_ft_rounds", "zero_shot_test_batch"])

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
            if type=='ft':
                noft = group_df["balanced_acc_no_ft"].values
                acc_noft_means.append(np.mean(noft))
                acc_noft_stds.append(np.std(noft))

            n = len(num_prev_ft_rounds)

            summary_subject = {
                "subject_id": subject_id,
                "subj_acc_means": np.array(acc_means) * 100.0,
                "subjs_acc_std": np.array(acc_stds) * 100.0,
                "subjs_acc_means_noft": (
                    np.array(acc_noft_means) * 100.0 if type=='ft' else np.full(n, np.nan)
                ),
                "subjs_acc_std_noft": (
                    np.array(acc_noft_stds) * 100.0 if type=='ft' else np.full(n, np.nan)
                ),
                "num_prev_ft_rounds": np.array(num_prev_ft_rounds),
                "lr": lr,
                "scheduler": sched_name,
                "num_ft_epochs": num_epochs,
            }

        summary_subject = _ensure_sorted_by_x(summary_subject)
        summary_condition_across_subjects.append(summary_subject)

    return summary_condition_across_subjects


def summarize_subject_table(summary_condition_across_subjects, ft_id, condition):
    """
    Builds a table with per-subject mean/std for each prev_ft_round, for:
      - Fine Tuning (FT) and
      - Zero Shot (on InterSessionModel) (no-FT baseline)
    Also includes global averages across subjects.
    """
    # assume identical x across subjects after sorting
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

    # Global averages across subjects at each round (mean across subject means; std across subject means)
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

    df_global = (
    df.groupby(["condition", "ft_id", "prev_ft_rounds"], as_index=False)[["FT_mean", "NOFT_mean"]].apply(_global_stats))

    return df, df_global

def plot_single_ft_strategy(
    summary_condition,
    condition,
    show_no_ft=True,
    title=None,
    save_path=None,
    ):
    """
    Like before: 2 panels
      - left: per-subject (same color per subject; solid=FT, dashed=Zero Shot on InterSessionModel)
      - right: average across subjects
    Also returns lr, num_epochs, scheduler (taken from first subject).
    """
    lr = summary_condition[0]["lr"]
    num_epochs = summary_condition[0]["num_ft_epochs"]
    scheduler = summary_condition[0]["scheduler"]

    all_means = np.stack([s["subj_acc_means"] for s in summary_condition], axis=0)
    avg_mean  = all_means.mean(axis=0)
    avg_std   = np.std(all_means, axis=0)

    if show_no_ft:
        all_means_noft = np.stack([s["subjs_acc_means_noft"] for s in summary_condition], axis=0)
        avg_mean_noft  = all_means_noft.mean(axis=0)
        avg_std_noft   = np.std(all_means_noft, axis=0)

    x = summary_condition[0]["num_prev_ft_rounds"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)

    cmap = plt.get_cmap("tab10")
    subject_colors = {
        subj["subject_id"]: cmap(i % 10)
        for i, subj in enumerate(summary_condition)
    }

    for subj in summary_condition:
        color = subject_colors[subj["subject_id"]]
        ax_left.errorbar(
            subj["num_prev_ft_rounds"],
            subj["subj_acc_means"],
            yerr=subj["subjs_acc_std"],
            marker="o",
            linestyle="-",
            color=color,
            capsize=4,
            alpha=0.9,
        )
        if show_no_ft:
            ax_left.errorbar(
                subj["num_prev_ft_rounds"],
                subj["subjs_acc_means_noft"],
                yerr=subj["subjs_acc_std_noft"],
                marker="o",
                linestyle="--",
                color=color,
                capsize=4,
                alpha=0.65,
            )

    ax_left.set_xlabel("Previous Training rounds", fontsize=11)
    ax_left.set_ylabel("Balanced accuracy (%)", fontsize=11)
    ax_left.set_title("Per-subject performance", fontsize=12)
    ax_left.grid(True, linestyle="--", alpha=0.5)
    
    ax_left.set_yticks(np.arange(0, 101, 10))
    ax_left.set_ylim(40, 90)
    ax_left.set_xticks(np.unique(x))

    ax_right.errorbar(
        x, avg_mean, yerr=avg_std,
        marker="o", linestyle="-", capsize=5, linewidth=2.0,
        label="Fine Tuning"
    )
    if show_no_ft:
        ax_right.errorbar(
            x, avg_mean_noft, yerr=avg_std_noft,
            marker="o", linestyle="--", capsize=5, linewidth=2.0,
            label="Zero Shot (on InterSessionModel)"
        )

    ax_right.set_xlabel("Previous Training rounds", fontsize=11)
    ax_right.set_title("Average across subjects", fontsize=12)
    ax_right.grid(True, linestyle="--", alpha=0.5)
    
    ax_right.set_yticks(np.arange(0, 101, 10))
    ax_right.set_ylim(40, 90)
    ax_right.set_xticks(np.unique(x))
    ax_right.legend(frameon=False, ncol=1)

    legend_elements = [Line2D([0], [0], linestyle="-", marker="o", label="Fine Tuning")]
    if show_no_ft:
        legend_elements.append(Line2D([0], [0], linestyle="--", marker="o", label="Zero Shot (on InterSessionModel)"))

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=len(legend_elements),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.92),
    )

    nice_title = f"{condition} — FT strategy: lr={_fmt_sci(lr)}, MAXep={num_epochs}, {scheduler}"
    if title is not None:
        nice_title += f"\n{title}"
    fig.suptitle(nice_title, fontsize=14, y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {"lr": lr, "num_epochs": num_epochs, "scheduler": scheduler}



def plot_single_ft_strategy_and_bs(
    summary_condition,
    summary_condition_baseline,
    condition,
    show_no_ft=False,
    title=None,
    save_path=None,
):
    # --- hyperparams (available to print/use) ---
    lr_ft = summary_condition[0]["lr"]
    num_epochs_ft = summary_condition[0]["num_ft_epochs"]
    scheduler_ft = summary_condition[0]["scheduler"]

    lr_bs = summary_condition_baseline[0]["lr"]
    num_epochs_bs = summary_condition_baseline[0]["num_ft_epochs"]
    scheduler_bs = summary_condition_baseline[0]["scheduler"]

    # --- map by subject for overlay ---
    ft_by_subj = {s["subject_id"]: s for s in summary_condition}
    bl_by_subj = {s["subject_id"]: s for s in summary_condition_baseline}

    # --- x ticks (check they match) ---
    x_ft = np.asarray(summary_condition[0]["num_prev_ft_rounds"])
    x_bl = np.asarray(summary_condition_baseline[0]["num_prev_ft_rounds"])

    if not np.array_equal(x_ft, x_bl):
        # safest: use intersection to avoid misaligned points
        x_common = np.intersect1d(x_ft, x_bl)
    else:
        x_common = x_ft

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

    # --- compute averages (FT) on x_common ---
    ft_means = []
    ft_noft_means = []
    for s in summary_condition:
        _, m, _, m_noft, _ = _slice_to_x(s, x_common)
        ft_means.append(m)
        if show_no_ft:
            ft_noft_means.append(m_noft)

    ft_means = np.stack(ft_means, axis=0)
    avg_mean_ft = ft_means.mean(axis=0)
    avg_std_ft  = np.std(ft_means, axis=0)

    if show_no_ft:
        ft_noft_means = np.stack(ft_noft_means, axis=0)
        avg_mean_noft = ft_noft_means.mean(axis=0)
        avg_std_noft  = np.std(ft_noft_means, axis=0)

    # --- baseline averages (train from scratch) on x_common ---
    bl_means = []
    for s in summary_condition_baseline:
        _, m, _, _, _ = _slice_to_x(s, x_common)
        bl_means.append(m)

    bl_means = np.stack(bl_means, axis=0)
    avg_mean_bl = bl_means.mean(axis=0)
    avg_std_bl  = np.std(bl_means, axis=0)

    # --- assign one color per subject (consistent) ---
    subject_ids = [s["subject_id"] for s in summary_condition]
    cmap = plt.get_cmap("tab10")
    color_by_subject = {sid: cmap(i % 10) for i, sid in enumerate(subject_ids)}

    # ---- Figure ----
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)

    # ---- LEFT: per-subject curves ----
    for sid in subject_ids:
        subj_ft = ft_by_subj[sid]
        subj_bl = bl_by_subj.get(sid, None)
        c = color_by_subject[sid]

        x, m_ft, s_ft, _, _ = _slice_to_x(subj_ft, x_common)
        ax_left.errorbar(x, m_ft, yerr=s_ft, marker="o", linestyle="-",  color=c, capsize=4, alpha=0.9)

        if subj_bl is not None:
            x, m_bl, s_bl, _, _ = _slice_to_x(subj_bl, x_common)
            ax_left.errorbar(x, m_bl, yerr=s_bl, marker="o", linestyle="--", color=c, capsize=4, alpha=0.9)

    ax_left.set_xticks(x_common)
    ax_left.set_xlabel("Prev. FT rounds")
    ax_left.set_ylabel("Accuracy (%)")
    ax_left.set_title("Per-subject")
    ax_left.grid(True, linestyle="--", alpha=0.6)
    ax_left.set_ylim(0, 100)
    ax_left.set_yticks(np.arange(0, 101, 10))

    # ---- Shared legend (clean): subjects by color + styles by line ----
    subject_handles = [Line2D([0], [0], color=color_by_subject[s], lw=3, label=s) for s in subject_ids]
    style_handles = [
        Line2D([0], [0], color="black", lw=3, linestyle="-",  label="Fine Tuning"),
        Line2D([0], [0], color="black", lw=3, linestyle="--", label="Train From Scratch"),
    ]

    fig.legend(
        handles=subject_handles + style_handles,
        loc="upper center",
        ncol=min(len(subject_handles) + len(style_handles), 6),  # prevents overflow
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=11,
    )

    # ---- RIGHT: averages ----
    ax_right.errorbar(
        x_common,
        avg_mean_ft,
        yerr=avg_std_ft,
        marker="s",
        linestyle="-",
        color="black",
        capsize=5,
        linewidth=2.2,
        label=f"Fine Tuning (avg) | min={avg_mean_ft.min():.1f}, max={avg_mean_ft.max():.1f}",
    )

    if show_no_ft:
        ax_right.errorbar(
            x_common,
            avg_mean_noft,
            yerr=avg_std_noft,
            marker="o",
            linestyle="--",
            color="gray",
            capsize=5,
            linewidth=2.0,
            label=f"No Fine-Tune (Intersession) (avg) | min={avg_mean_noft.min():.1f}, max={avg_mean_noft.max():.1f}",
        )

    ax_right.errorbar(
        x_common,
        avg_mean_bl,
        yerr=avg_std_bl,
        marker="s",
        linestyle="--",
        color="black",
        capsize=5,
        linewidth=2.2,
        label=f"Train From Scratch (avg) | min={avg_mean_bl.min():.1f}, max={avg_mean_bl.max():.1f}",
    )

    ax_right.set_xticks(x_common)
    ax_right.set_xlabel("Prev. FT rounds")
    ax_right.set_title("Average across subjects")
    ax_right.set_yticks(np.arange(0, 101, 10))
    ax_right.grid(True, linestyle="--", alpha=0.6)
    ax_right.set_ylim(0, 100)
    ax_right.legend(frameon=False)

    # ---- Title ----
    supt = (
        f"{condition} — FT vs Train-from-scratch\n"
        f"FT: lr={_fmt_sci(lr_ft)}, ep={num_epochs_ft}, {scheduler_ft} | "
        f"Scratch: lr={_fmt_sci(lr_bs)}, ep={num_epochs_bs}, {scheduler_bs}"
    )
    if title:
        supt += f"\n{title}"
    fig.suptitle(supt, fontsize=14, y=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return (
        {"lr": lr_ft, "num_epochs": num_epochs_ft, "scheduler": scheduler_ft},
        {"lr": lr_bs, "num_epochs": num_epochs_bs, "scheduler": scheduler_bs},
    )


def plot_compare_ft_configs_avg(
    base_ft_model_folder,
    condition,
    subjects,
    ft_ids,
    model_base_id,
    show_no_ft=True,
    save_path=None,
    ):
    """
    Plot ONLY the average across subjects, comparing multiple ft_config_ids.
    Returns:
      - cfg_info: dict ft_id -> {"lr":..., "num_epochs":..., "scheduler":...}
      - df_subject: per-subject table (mean/std per round)
      - df_global: global table (mean/std across subjects per round)
    """
    cfg_info = {}
    per_ft_subject_tables = []
    per_ft_global_tables = []

    # Larger figure for multi-config comparison
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    for ft_id in ft_ids:
        summary = load_results(base_ft_model_folder, condition, subjects, ft_id, model_base_id)

        # hyperparams (identical across subjects by construction)
        lr = summary[0]["lr"]
        num_epochs = summary[0]["num_ft_epochs"]
        scheduler = summary[0]["scheduler"]
        cfg_info[ft_id] = {"lr": lr, "num_epochs": num_epochs, "scheduler": scheduler}

        # x axis (assume consistent across subjects)
        x = summary[0]["num_prev_ft_rounds"]

        # average across subjects
        all_means = np.stack([s["subj_acc_means"] for s in summary], axis=0)
        avg_mean = all_means.mean(axis=0)
        avg_std  = np.std(all_means, axis=0)

        # label includes scientific notation for lr
        lbl = f"{ft_id} (lr={_fmt_sci(lr)}, ep={num_epochs}, {scheduler})"

        ax.errorbar(
            x,
            avg_mean,
            yerr=avg_std,
            marker="o",
            linestyle="-",
            capsize=5,
            linewidth=2.0,
            label=lbl,
        )

        # optional baseline (no-FT / zero-shot intersession model)
        if show_no_ft:
            all_means_noft = np.stack([s["subjs_acc_means_noft"] for s in summary], axis=0)
            avg_mean_noft = all_means_noft.mean(axis=0)
            avg_std_noft  = np.std(all_means_noft, axis=0)

            ax.errorbar(
                x,
                avg_mean_noft,
                yerr=avg_std_noft,
                marker="x",
                linestyle="--",
                capsize=5,
                linewidth=1.8,
                label=f"{ft_id} — Zero Shot (on InterSessionModel)",
                alpha=0.85,
            )

        # tables
        df_subj, df_glob = summarize_subject_table(summary, ft_id=ft_id, condition=condition)
        per_ft_subject_tables.append(df_subj)
        per_ft_global_tables.append(df_glob)

    df_subject = pd.concat(per_ft_subject_tables, ignore_index=True)
    df_global  = pd.concat(per_ft_global_tables, ignore_index=True)

    # Plot styling
    ax.set_xlabel("Previous fine-tuning rounds", fontsize=12)
    ax.set_ylabel("Balanced accuracy (%)", fontsize=12)
    ax.set_xticks(np.unique(df_subject["prev_ft_rounds"].values))
    
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylim(40, 90)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Nicely structured title
    title = f"Fine-tuning strategy comparison — condition: {condition}"
    ax.set_title(title, fontsize=14, pad=12)

    # Legend across the top (one line if possible)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        ncol=2,  # increase if you have many ft_ids
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return cfg_info, df_subject, df_global


def print_tables(df_subject: pd.DataFrame, df_global: pd.DataFrame):
    """
    Prints:
      1) per-subject mean/std table (FT + ZS intersession)
      2) global mean/std (across subjects) per round
    """
    # Per-subject table
    df_subject_print = df_subject.copy()
    for c in ["FT_mean", "FT_std", "NOFT_mean", "NOFT_std"]:
        df_subject_print[c] = df_subject_print[c].map(lambda v: f"{v:6.2f}")

    print("\n==================== Per-subject (mean/std across folds) ====================")
    print(df_subject_print.sort_values(["condition", "ft_id", "subject", "prev_ft_rounds"]).to_string(index=False))

    # Global table
    df_global_print = df_global.copy()
    for c in ["FT_mean_global", "FT_std_global", "NOFT_mean_global", "NOFT_std_global"]:
        df_global_print[c] = df_global_print[c].map(lambda v: f"{v:6.2f}")

    print("\n==================== Global (across subjects) ====================")
    print(df_global_print.sort_values(["condition", "ft_id", "prev_ft_rounds"]).to_string(index=False))




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

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

def _minmaxavg(y):
    y = np.asarray(y, dtype=float)
    return float(np.min(y)), float(np.max(y)), float(np.mean(y))

def plot_per_subject_grid(
    ft_summary,
    scratch_summary,
    condition,
    show_no_ft=True,
    save_path=None,
    ):
    x, subject_ids, per_subj, info_ft, info_sc, avg = prepare_aligned(ft_summary, scratch_summary, show_no_ft)

    n_subj = len(subject_ids)
    ncols = 2
    nrows = int(np.ceil(n_subj / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    fs_title, fs_sub, fs_ax, fs_label, fs_tick, fs_box, fs_leg = 16, 12, 14, 14, 12, 11, 12

    for ax, sid in zip(axes, subject_ids):
        d = per_subj[sid]

        ax.errorbar(d["x"], d["ft_mean"], yerr=d["ft_std"],
                    marker="o", linestyle="-", capsize=4, linewidth=2.4)
        if show_no_ft:
            ax.errorbar(d["x"], d["nf_mean"], yerr=d["nf_std"],
                        marker="o", linestyle="--", capsize=4, linewidth=2.2)
        ax.errorbar(d["x"], d["sc_mean"], yerr=d["sc_std"],
                    marker="s", linestyle="-.", capsize=4, linewidth=2.4)

        ax.set_title(sid, fontsize=fs_ax)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xticks(x)
        ax.tick_params(axis="both", labelsize=fs_tick)

        ft_min, ft_max, ft_avg = _minmaxavg(d["ft_mean"])
        sc_min, sc_max, sc_avg = _minmaxavg(d["sc_mean"])
        lines = [f"FT:   min={ft_min:.1f}, max={ft_max:.1f}, avg={ft_avg:.1f}"]
        if show_no_ft:
            nf_min, nf_max, nf_avg = _minmaxavg(d["nf_mean"])
            lines.append(f"NoFT: min={nf_min:.1f}, max={nf_max:.1f}, avg={nf_avg:.1f}")
        lines.append(f"Scr:  min={sc_min:.1f}, max={sc_max:.1f}, avg={sc_avg:.1f}")

        ax.text(
            0.25, 0.02,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=fs_box,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, linewidth=0.6),
        )

    for ax in axes[len(subject_ids):]:
        ax.axis("off")

    fig.supxlabel("Previous fine-tuning rounds", fontsize=fs_label)
    fig.supylabel("Balanced accuracy (%)", fontsize=fs_label)

    fig.suptitle(f"{_cond_title(condition)} — Per-subject comparison", fontsize=fs_title, y=0.98)
    subtitle = (
        f"FT: lr={_fmt_sci(info_ft['lr'])}, ep={info_ft['num_epochs']} | "
        f"Scratch: lr={_fmt_sci(info_sc['lr'])}, ep={info_sc['num_epochs']}"
    )
    fig.text(0.5, 0.945, subtitle, ha="center", va="center", fontsize=fs_sub)

    style_handles = [Line2D([0], [0], color="black", lw=2.6, linestyle="-",  marker="o", label="Fine Tuning")]
    if show_no_ft:
        style_handles.append(Line2D([0], [0], color="black", lw=2.6, linestyle="--", marker="o", label="No Fine Tuning"))
    style_handles.append(Line2D([0], [0], color="black", lw=2.6, linestyle="-.", marker="s", label="Train From Scratch"))

    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=len(style_handles),
        frameon=False,
        fontsize=fs_leg,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=0.8,
        labelspacing=0.2,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.10, top=0.84, wspace=0.12, hspace=0.22)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return info_ft, info_sc

def prepare_aligned(ft_summary, scratch_summary, show_no_ft=True):
    """
    Align FT and Scratch summaries on a common x and return:
      - x
      - subject_ids
      - per_subj dict with aligned arrays
      - info_ft, info_scratch
      - avg curves (mean/std across subjects) for ft/noft/scratch
    """
    # hyperparams once
    info_ft = {"lr": ft_summary[0]["lr"], "num_epochs": ft_summary[0]["num_ft_epochs"], "scheduler": ft_summary[0]["scheduler"]}
    info_sc = {"lr": scratch_summary[0]["lr"], "num_epochs": scratch_summary[0]["num_ft_epochs"], "scheduler": scratch_summary[0]["scheduler"]}

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

    # average curves across subjects
    ft_stack = np.stack([per_subj[sid]["ft_mean"] for sid in subject_ids], axis=0)
    sc_stack = np.stack([per_subj[sid]["sc_mean"] for sid in subject_ids], axis=0)
    ft_mean = ft_stack.mean(axis=0)
    ft_std  = np.std(ft_stack, axis=0)
    sc_mean = sc_stack.mean(axis=0)
    sc_std  = np.std(sc_stack, axis=0)

    nf_mean = nf_std = None
    if show_no_ft:
        nf_stack = np.stack([per_subj[sid]["nf_mean"] for sid in subject_ids], axis=0)
        nf_mean = nf_stack.mean(axis=0)
        nf_std  = np.std(nf_stack, axis=0)

    avg = {
        "ft_mean": ft_mean, "ft_std": ft_std,
        "nf_mean": nf_mean, "nf_std": nf_std,
        "sc_mean": sc_mean, "sc_std": sc_std,
    }

    return x, subject_ids, per_subj, info_ft, info_sc, avg

def plot_average_across_subjects(
    ft_summary,
    scratch_summary,
    condition,
    show_no_ft=True,
    save_path=None,
):
    # helpers already in your code:
    # prepare_aligned, _minmaxavg, _cond_title, _fmt_sci

    x, subject_ids, per_subj, info_ft, info_sc, avg = prepare_aligned(
        ft_summary, scratch_summary, show_no_ft
    )

    # ---- metrics ----
    ft_min, ft_max, ft_avg = _minmaxavg(avg["ft_mean"])
    sc_min, sc_max, sc_avg = _minmaxavg(avg["sc_mean"])

    lines = [
        f"Fine Tuning:     min={ft_min:.1f}  max={ft_max:.1f}  avg={ft_avg:.1f}",
    ]
    if show_no_ft:
        nf_min, nf_max, nf_avg = _minmaxavg(avg["nf_mean"])
        lines.append(f"No Fine Tuning:  min={nf_min:.1f}  max={nf_max:.1f}  avg={nf_avg:.1f}")
    lines.append(f"Train Scratch:   min={sc_min:.1f}  max={sc_max:.1f}  avg={sc_avg:.1f}")

    # ---- FIGURE ----
    fs_title = 18
    fs_label = 16
    fs_tick = 13
    fs_leg = 10
    fs_box = 13

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

    # curves (short legend labels; metrics are in the textbox)
    ax.errorbar(
        x, avg["ft_mean"], yerr=avg["ft_std"],
        marker="o", linestyle="-", capsize=5, linewidth=2,
        label=f"Fine Tuning | lr={_fmt_sci(info_ft['lr'])}, ep={info_ft['num_epochs']}",
    )

    if show_no_ft:
        ax.errorbar(
            x, avg["nf_mean"], yerr=avg["nf_std"],
            marker="o", linestyle="--", capsize=4, linewidth=2,
            label="No Fine Tuning",
        )

    ax.errorbar(
        x, avg["sc_mean"], yerr=avg["sc_std"],
        marker="s", linestyle="-.", capsize=5, linewidth=2,
        label=f"From Scratch | lr={_fmt_sci(info_sc['lr'])}, ep={info_sc['num_epochs']}",
    )

    # axes styling
    ax.set_title(f"{_cond_title(condition)} — Average across subjects", fontsize=fs_title, pad=10)
    ax.set_xlabel("Previous fine-tuning rounds", fontsize=fs_label)
    ax.set_ylabel("Balanced accuracy (%)", fontsize=fs_label)
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=fs_tick)

    # legend: TOP-LEFT INSIDE
    ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=fs_leg,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=2.0,
        handletextpad=0.5,
    )

    # metrics box: BOTTOM-CENTER INSIDE
    # (use ha="center" and x=0.5 to center it)
    ax.text(
        0.50, 0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=fs_box,
        va="bottom",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, linewidth=0.6),
    )

    # # subtitle with hyperparams (once)
    # subtitle = (
    #     f"FT: lr={_fmt_sci(info_ft['lr'])}, ep={info_ft['num_epochs']} | "
    #     f"Scratch: lr={_fmt_sci(info_sc['lr'])}, ep={info_sc['num_epochs']}"
    # )
    # fig.text(0.5, 0.965, subtitle, ha="center", va="center", fontsize=13)

    # manual layout (avoid tight_layout)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14, top=0.86)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return info_ft, info_sc

if __name__ == "__main__":
    # for condition in conditions:
    #     cfg_info, df_subject, df_global = plot_compare_ft_configs_avg(base_ft_model_folder=base_ft_model_folder,
    #         condition=condition,
    #         subjects=subjects,
    #         ft_ids=ft_configs_ids,
    #         model_base_id=model_base_id,
    #         show_no_ft=False,
    #         save_path=fig_path / f"compare_ft_configs_{condition}.png")



    ft_id = "ft_config_0"
    bs_id = "bs_config_0"
    for condition in conditions:
        summary_ft = load_results(base_ft_model_folder, condition, subjects, ft_id, model_base_id, type='ft')

        info = plot_single_ft_strategy(
            summary_condition=summary_ft,
            condition=condition,
            show_no_ft=True,
            title=f"ft_id={ft_id}_condition:{condition}",
            save_path=fig_path / f"ft_{ft_id}_{condition}.png",
        )
        print("\nFine-Tuning hyperparams:", info)


        # Load also summary for the baseline
        summary_baseline = load_results(baseline_model_folder, condition, subjects, bs_id, model_base_id, type='baseline')
        plot_single_ft_strategy_and_bs(summary_ft,summary_baseline,condition,show_no_ft=True, title=None,save_path=fig_path/f'comparison_{condition}')


        print(f"================== {condition}===============")
        # info_ft, info_scratch, summary_stats = plot_subject_grid_only(
        # summary_ft=summary_ft,
        # summary_from_scratch=summary_baseline,
        # condition=condition,
        # save_dir=fig_path,
        # fig_prefix=f"ft_vs_scratch_{condition}")
        # print(info_ft, info_scratch)

        # print(summary_stats)
        plot_per_subject_grid(
            ft_summary=summary_ft,
            scratch_summary=summary_baseline,
            condition=condition,
            show_no_ft=True,
            save_path=fig_path / f"grid_{condition}.png",
        )

        plot_average_across_subjects(
            ft_summary=summary_ft,
            scratch_summary=summary_baseline,
            condition=condition,
            show_no_ft=True,
            save_path=fig_path / f"avg_{condition}.png",
        )