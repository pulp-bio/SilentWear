"""
This file can be used to summarize the results of the "Global" and "Inter-Session Experiments".
It will read the results for **all** models which have been trained. 
We assume that you have runned the "Global" and "Inter-Session experiments" as described in the README.md file. 
It is useful when you want to generate summaries for different variants (e.g: same model, trained under different seed)

"""
import os
import pandas as pd
from pathlib import Path
import sys
import json
import ast
import numpy as np
import sys
from pathlib import Path
project_root = Path().resolve()
ARTIFACTS_DIR = Path(os.environ.get('SILENTWEAR_ARTIFACTS_DIR', project_root / 'artifacts'))
.parent.parent
sys.path.insert(0, str(project_root))
from utils.III_results_analysis.utils import *


################## USER EDITABLE PART ##########################################################
model_results_dire = ARTIFACTS_DIR / 'models'                          # The directory where your models were saved
res_save_folder = ARTIFACTS_DIR                     # The directory where you want to save csv files with results
fig_save_folder = ARTIFACTS_DIR / 'figures' / 'figures'                            # The directory where figures will be saved
#fig_save_folder.mkdir(parents=True, exist_ok=True)
experiment_to_analyze = "global"                # this can be #global or inter_session
subjects_to_consider = ["S01", "S02", "S03", "S04"]
conditions_to_consider = ["silent", "vocalized"]
model_to_select = "speechnet_base"                  # supported: speechnet_base, random_forest (or whatever you added in models_factory.py)
win_size_ms_to_consider= 1400
plot_acc_per_word = True
############################################################################################



if __name__=='__main__':
    summary_df = load_all_results(model_results_dire/experiment_to_analyze, subjects_to_consider, conditions_to_consider)

    # Filter by model name 
    model_df = summary_df[summary_df["model_name"] == model_to_select]

    
    print("\n\n================== MODEL COMPARISON REPORT ==================\n")
    dfs_shared = {
        sig: group for sig, group in model_df.groupby("run_cfg_signature_exact")if len(group) > 1
    }

    for key_id in range(len(dfs_shared.keys())):
        curr_key = list(dfs_shared.keys())[key_id]
        curr_df = dfs_shared[curr_key]
        print(f"\n=== Model Variant {key_id+1}| sig: {curr_key} ===\n")

        if model_to_select!='random_forest':
            print("Architecture blocks:")
            blocks_config = curr_df["run_cfg"].iloc[0]["model_cfg"]["model"]["kwargs"]['blocks_config']
            for curr_block in blocks_config:
                print(curr_block)


        # check if results were 
        seed_sigs = np.sort(list(curr_df["run_cfg_signature_seeds"].unique()))
        silent_accs_seeds = []
        silent_accs_vals_seeds = []
        vocalized_accs_vals_seeds = []
        vocalized_accs_seeds = []
        for sig_seed_id in range(len(seed_sigs)):

            model_seed_df = curr_df[curr_df["run_cfg_signature_seeds"] == seed_sigs[sig_seed_id]]


            model_win = model_seed_df[model_seed_df['win_size_ms']==win_size_ms_to_consider]
            print(model_win.keys())
            
            silent_accs = model_win[model_win['condition'] == 'silent']['balanced_acc_mean'].values
            silent_accs_vals = model_win[model_win['condition'] == 'silent']['balanced_acc_vals'].values
            vocalized_accs = model_win[model_win['condition'] == 'vocalized']['balanced_acc_mean'].values
            vocalized_accs_vals = model_win[model_win['condition'] == 'vocalized']['balanced_acc_vals'].values

            if len(vocalized_accs)==len(subjects_to_consider) and len(silent_accs)==len(subjects_to_consider):
                print("----------------------------")
                print(f"Seed variant {sig_seed_id} | seed sig: {seed_sigs[sig_seed_id]}")
                print("Seeds set to:", model_seed_df["run_cfg"].iloc[0].get("seeds", "default"))
                silent_accs_seeds.append(silent_accs)
                silent_accs_vals_seeds.append(silent_accs_vals)
                #print(f"Silent ACC | avg across subjects: {np.mean(silent_accs):.2f} ± {np.std(silent_accs):.2f}")
                print(f"Per subject silent ACCs: {silent_accs}")
                vocalized_accs_seeds.append(vocalized_accs) 
                vocalized_accs_vals_seeds.append(vocalized_accs_vals)

                
                #print(f"Vocalized ACC | avg across subjects: {np.mean(vocalized_accs):.2f} ± {np.std(vocalized_accs):.2f}")
                print(f"Per subject vocalized ACCs: {vocalized_accs}")

                fig_save_path = fig_save_folder / f"{experiment_to_analyze}_{model_to_select}_VAR{key_id+1}_SEED{sig_seed_id}_WIN{win_size_ms_to_consider}_acc_per_word_s2.png"
                if plot_acc_per_word:
                    plot_subject_word_accuracy_grid_from_summary(model_win, "vocalized", "silent", title_extras=f"{model_to_select}| WIN_{win_size_ms_to_consider} |seed2", save_path=fig_save_path)
                    plt.show()
                print("----------------------------")
        # summarize across seeds
        # First, average per subject across seeds
        print("======\n")
        silent_accs_seeds = np.array(silent_accs_seeds)   # shape
        vocalized_accs_seeds = np.array(vocalized_accs_seeds)

        silent_accs_vals_seeds = np.array(silent_accs_vals_seeds)   # shape
        vocalized_accs_vals_seeds = np.array(vocalized_accs_vals_seeds)
       

        base_name = f"{experiment_to_analyze}_{model_to_select}_VAR{key_id+1}_WIN{win_size_ms_to_consider}"
        silent_csv   = res_save_folder / f"{base_name}_silent_seed_report.csv"
        vocalized_csv= res_save_folder / f"{base_name}_vocalized_seed_report.csv"
        
        save_per_condition_seed_report_csv(
            accs_seeds_raw=vocalized_accs_seeds,
            accs_vals_seeds_raw=silent_accs_vals_seeds,
            subjects=subjects_to_consider,
            condition_name="silent",
            out_csv_path=silent_csv,
            values_are_fraction=True,
        )
        save_per_condition_seed_report_csv(
            accs_seeds_raw=vocalized_accs_seeds,
            accs_vals_seeds_raw=vocalized_accs_vals_seeds,
            subjects=subjects_to_consider,
            condition_name="vocalized",
            out_csv_path=vocalized_csv,
            values_are_fraction=True,
        )

        mean_silent_accs_per_subject = np.mean(silent_accs_seeds, axis=0) * 100
        mean_silent_accs_per_subject = np.round(mean_silent_accs_per_subject, 1)

        print(
            "Average Silent ACCs per subject across seeds (%):",
            mean_silent_accs_per_subject,
            "shape",
            mean_silent_accs_per_subject.shape,
        )

        mean_silent_acc_all = np.round(np.mean(mean_silent_accs_per_subject), 1)
        std_silent_acc_all  = np.round(np.std(mean_silent_accs_per_subject), 1)

        print(
            f"Overall Silent ACC averaged across subjects and seeds: "
            f"{mean_silent_acc_all} ± {std_silent_acc_all} %"
        )


        mean_vocalized_accs_per_subject = np.mean(vocalized_accs_seeds, axis=0) * 100
        mean_vocalized_accs_per_subject = np.round(mean_vocalized_accs_per_subject, 1)

        print(
            "Average Vocalized ACCs per subject across seeds (%):",
            mean_vocalized_accs_per_subject,
            "shape",
            mean_vocalized_accs_per_subject.shape,
        )

        mean_vocalized_acc_all = np.round(np.mean(mean_vocalized_accs_per_subject), 1)
        std_vocalized_acc_all  = np.round(np.std(mean_vocalized_accs_per_subject), 1)

        print(
            f"Overall Vocalized ACC averaged across subjects and seeds: "
            f"{mean_vocalized_acc_all} ± {std_vocalized_acc_all} %"
        )


# if __name__=='__main__':
#     summary_df = load_all_results(model_results_dire/experiment_to_analyze, subjects_to_consider, conditions_to_consider)

#     # Filter by model
#     model_df = summary_df[summary_df["model_name"] == model_to_select]
#     # The same model can have different variants (e.g., different hyperparams, filters, kernel sizes) etc
    
#     print("\n\n================== MODEL COMPARISON REPORT ==================\n")
#     dfs_shared = {
#         sig: group
#         for sig, group in model_df.groupby("run_cfg_signature_exact")
#         if len(group) > 1
#     }

#     for key_id in range(len(dfs_shared.keys())):
#         scheduler = "None"
#         curr_key = list(dfs_shared.keys())[key_id]
#         curr_df = dfs_shared[curr_key]
#         print(f"\n=== Model Variant {key_id+1}| sig: {curr_key} ===\n")

#         if model_to_select!='random_forest':
#             print("Architecture blocks:")
#             blocks_config = curr_df["run_cfg"].iloc[0]["model_cfg"]["model"]["kwargs"]['blocks_config']
#             for curr_block in blocks_config:
#                 print(curr_block)


#             # Here, we have selected the same Model CFG.
#             # Just make sure the config are actually uniques
            
#             ignore_keys = [
#                 ["base_cfg", "data"],                # note: here the strucuure is parent key -> child key
#                 ["base_cfg", "paths"], 
#                 ["base_cfg", "subject"],
#                 ["base_cfg", "condition"],           # same architecture and settings for both silent and vocalized experiments           
#                 ["subject"], 
#                 ["condition"],
#                 ["model_cfg", "model", "kwargs", "train_cfg", "scheduler"],             #['model_cfg']["model"]["kwargs"]["train_cfg"]['']
#                 ["seeds"]          
#             ]
#             canon = curr_df["run_cfg"].apply(lambda x: normalize_and_canonicalize(x, ignore_keys=ignore_keys))
#             unique_run_cfgs = canon.unique()
            

#             print("Unique run_cfg (after dropping data/subject):", len(canon))
            

#             if len(unique_run_cfgs)>1:
#                 print("[ERROR]: multiple run_cfg found!")
#                 print(unique_run_cfgs)
#                 sys.exit()
#             else:
#                 run_cfg = unique_run_cfgs[0]
#                 print("Run CFG:", run_cfg)
                
#             break
#             if run_cfg["train_cfg"].get("scheduler") is not None:
#                 scheduler = run_cfg["train_cfg"]["scheduler"]
#                 print("Scheduler:", run_cfg["train_cfg"]["scheduler"])

#         break
#         # check if results were 
#         seed_sigs = np.sort(list(curr_df["run_cfg_signature_seeds"].unique()))
#         silent_accs_seeds = []
#         vocalized_accs_seeds = []
#         for sig_seed_id in range(len(seed_sigs)):

#             model_seed_df = curr_df[curr_df["run_cfg_signature_seeds"] == seed_sigs[sig_seed_id]]


#             model_win = model_seed_df[model_seed_df['win_size_ms']==win_size_ms_to_consider]

#             silent_accs = model_win[model_win['condition'] == 'silent']['balanced_acc_mean'].values
#             vocalized_accs = model_win[model_win['condition'] == 'vocalized']['balanced_acc_mean'].values

#             if len(vocalized_accs)==len(subjects_to_consider) and len(silent_accs)==len(subjects_to_consider):
#                 print("----------------------------")
#                 print(f"Seed variant {sig_seed_id} | seed sig: {seed_sigs[sig_seed_id]}")
#                 print("Seeds set to:", model_seed_df["run_cfg"].iloc[0].get("seeds", "default"))
#                 silent_accs_seeds.append(silent_accs)
#                 #print(f"Silent ACC | avg across subjects: {np.mean(silent_accs):.2f} ± {np.std(silent_accs):.2f}")
#                 print(f"Per subject silent ACCs: {silent_accs}")
#                 vocalized_accs_seeds.append(vocalized_accs) 
#                 #print(f"Vocalized ACC | avg across subjects: {np.mean(vocalized_accs):.2f} ± {np.std(vocalized_accs):.2f}")
#                 print(f"Per subject vocalized ACCs: {vocalized_accs}")

#                 fig_save_path = fig_save_folder / f"{experiment_to_analyze}_{model_to_select}_VAR{key_id+1}_SEED{sig_seed_id}_WIN{win_size_ms_to_consider}_acc_per_word.png"
#                 if plot_acc_per_word:
#                     title = f"{model_to_select}| VAR_{key_id+1}| SEED_{sig_seed_id}| WIN_{win_size_ms_to_consider}ms"
#                     if scheduler!="None":
#                         title += f"| Scheduler: {scheduler}"
#                     plot_subject_word_accuracy_grid_from_summary(model_win, "vocalized", "silent", title_extras=title, save_path=fig_save_path)
#                     plt.show()
#                 print("----------------------------")
#         # summarize across seeds
#         # First, average per subject across seeds

#         silent_accs_seeds = np.array(silent_accs_seeds)   # shape
#         vocalized_accs_seeds = np.array(vocalized_accs_seeds)

#         mean_silent_accs_per_subject = np.mean(silent_accs_seeds, axis=0) * 100
#         mean_silent_accs_per_subject = np.round(mean_silent_accs_per_subject, 1)

#         print(
#             "Average Silent ACCs per subject across seeds (%):",
#             mean_silent_accs_per_subject,
#             "shape",
#             mean_silent_accs_per_subject.shape,
#         )

#         mean_silent_acc_all = np.round(np.mean(mean_silent_accs_per_subject), 1)
#         std_silent_acc_all  = np.round(np.std(mean_silent_accs_per_subject), 1)

#         print(
#             f"Overall Silent ACC averaged across subjects and seeds: "
#             f"{mean_silent_acc_all} ± {std_silent_acc_all} %"
#         )


#         mean_vocalized_accs_per_subject = np.mean(vocalized_accs_seeds, axis=0) * 100
#         mean_vocalized_accs_per_subject = np.round(mean_vocalized_accs_per_subject, 1)

#         print(
#             "Average Vocalized ACCs per subject across seeds (%):",
#             mean_vocalized_accs_per_subject,
#             "shape",
#             mean_vocalized_accs_per_subject.shape,
#         )

#         mean_vocalized_acc_all = np.round(np.mean(mean_vocalized_accs_per_subject), 1)
#         std_vocalized_acc_all  = np.round(np.std(mean_vocalized_accs_per_subject), 1)

#         print(
#             f"Overall Vocalized ACC averaged across subjects and seeds: "
#             f"{mean_vocalized_acc_all} ± {std_vocalized_acc_all} %"
#         )

        



