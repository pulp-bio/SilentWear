"""
This script is used to run Global Models

Specifically:
Consider all data collected for the current subject (all acquisition sessions)
Run 5-fold CV, based on the CV setting defined in the config file. 

Each fold contains data from all SESSIONS.
"""

from pathlib import Path
import sys
import yaml
import json
import pandas as pd
import numpy as np
from Model_Master import Model_Master
from models.seeds import TORCH_MANUAL_SEED, RANDOM_SEED, RGN_SEED
from utils.general_utils import *
from offline_experiments.general_utils import *
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class Global_Model_Trainer:
    def __init__(self, 
                base_config, 
                model_config) -> None:
        pass
        self.base_config = base_config
        self.model_config = model_config
        self.model_master = None


        # Get Subject-ID - IDs
        self.sub_id = self.base_config['data']['subject_id']            # this is either a string or a list
        if type(self.sub_id) == str:
            self.all_subjects_models = False
        elif type(self.sub_id) == list:
            self.all_subjects_models = True
        else:
            print(f"Check subject ids in config file")
            sys.exit()

        self.condition = self.base_config['condition']                  # silent or vocalized or both
        self.main_dire = self.base_config['data']['data_directory']
        self.main_model_dire = self.base_config['data']['models_main_directory']
        self.model_name = self.model_config['model']['name']
        print("Requetsed model:", self.model_name)
        self.window_size_ms = int(self.base_config['window']['window_size_s'] * 1000)
        self.include_rest = self.base_config['experiment']['include_rest']
        self.model_kind = self.model_config['model']['kind']
        self.model_dire = self.create_saving_directory()

        # check if data directories exist.
        self.check_data_directory() 

        self.df = pd.DataFrame()

        self.cv_summaries = []
        
    def create_saving_directory(self):
        # Function to create saving directory for models and utils  
        # models will be saved under <models> / <global> / <SUB_ID > / <condition> / <model_name> / <model_run>

        # Check if one or multiple subjects
        if self.all_subjects_models == False:
            print(f"Data will be saved into subject-specific directory ({self.sub_id})")
            model_parent_dire = Path(self.main_model_dire) / Path(f"models/global/{self.sub_id}/{self.condition}/{self.model_name}")
        else:
            print(f"Data will be saved into global directory")
            model_parent_dire = Path(self.main_model_dire) / Path(f"models/global/all_subjects/{self.condition}/{self.model_name}")

        if model_parent_dire.exists() == False:
            model_parent_dire.mkdir(parents=True)
            print("Created directory", model_parent_dire)
        #check run id to append 

        model_id_base = 1
        model_dire_okay = False
        while not model_dire_okay:
            model_dire = model_parent_dire/f"model_{model_id_base}"
            if model_dire.exists():
                model_id_base+=1
            else:
                model_dire.mkdir()
                print("Models will be saved under:", model_dire)
                model_dire_okay = True

        return model_dire

    def check_data_directory(self):
        # Function to check if data directory exist. 
        self.data_dire_proc = []
        if self.all_subjects_models == False:
            # check if we want to train with silent - vocalized or both
            if self.condition != "voc_and_silent":
                self.data_dire_proc.append(self.main_dire / Path(f"{self.base_config['paths']['win_and_feats']}/{self.sub_id}/{self.condition}/WIN_{self.window_size_ms}"))
            else:
                self.data_dire_proc.append(self.main_dire / Path(f"{self.base_config['paths']['win_and_feats']}/{self.sub_id}/silent/WIN_{self.window_size_ms}"))
                self.data_dire_proc.append(self.main_dire / Path(f"{self.base_config['paths']['win_and_feats']}/{self.sub_id}/vocalized/WIN_{self.window_size_ms}"))

        elif self.all_subjects_models == True:
            for curr_sub_id in self.sub_id:
                if self.condition != "voc_and_silent":
                    self.data_dire_proc.append(self.main_dire / Path(f"{self.base_config['paths']['win_and_feats']}/{curr_sub_id}/{self.condition}/WIN_{self.window_size_ms}"))
                else:
                    self.data_dire_proc.append(self.main_dire / Path(f"{self.base_config['paths']['win_and_feats']}/{curr_sub_id}/silent/WIN_{self.window_size_ms}"))
                    self.data_dire_proc.append(self.main_dire / Path(f"{self.base_config['paths']['win_and_feats']}/{curr_sub_id}/vocalized/WIN_{self.window_size_ms}"))
        for data_dire_proc in self.data_dire_proc:        
            if data_dire_proc.exists() == False:
                print("Data directory:", self.data_dire_proc, "does not exist, exist")
                sys.exit()


    def save_run_cfg(self):
        # Function to save a json file with main settings of the current run
        run_cfg_dict = {
            "condition": self.condition,
            "experiment_type": "global",


            "experimental_settings": {
                "window_size_ms": self.window_size_ms, 
                "include_rest": self.base_config["experiment"]["include_rest"],
                "cv_type" : self.base_config['cv']
            },
            
            "model_cfg": self.model_config,
            "base_cfg" : self.base_config, 
            "seeds" : {
                "torch_manual_seed": TORCH_MANUAL_SEED,
                "random_seed": RANDOM_SEED,
                "rgn_seed": RGN_SEED,
            }
        }

        with open(self.model_dire / "run_cfg.json", "w") as f:
            json.dump(
                run_cfg_dict,
                f,
                indent=4,          # pretty formatting
                sort_keys=True     # optional: alphabetical order
            )

    def run_cv(self):
        cv_cfg = self.base_config.get("cv", {})
        mode = cv_cfg.get("mode")
        print("MODE SET:", mode)
        n_splits = int(cv_cfg.get("n_splits", 5))
        val_size = float(cv_cfg.get("val_size", 0.3))
        seed = int(self.base_config["experiment"]["seed"])

        # optional: remove rest once, consistently
        df = self.df
        #print(df.columns)
        if not self.include_rest:
            df = df[df["Label_str"] != "rest"].copy()

        print_dataset_summary_statistics(df)

        if mode == "group_kfold":
            return self._cv_groupkfold(df, n_splits=n_splits, val_size=val_size, seed=seed)
        elif mode == "stratified_kfold":
            return self._stratified_kfold(df, n_splits=n_splits, val_size=val_size, seed=seed)
        
        elif mode == "leave_one_batch_out":
            return self._cv_leave_one_batch_out(df, val_size=val_size, seed=seed)

        else:
            raise ValueError(f"Unknown cv.mode='{mode}' (use 'kfold' or 'leave_one_batch_out')")
    
    def _stratified_kfold(self, df, n_splits=5, val_size=0.3, seed=0):
        """Stratified K-Fold: balances class distribution across folds but may mix samples from the same batch/session between train and test."""
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

        # Labels for stratification
        y = df["Label_int"]  

        self.cv_summaries = []

        for fold_id, (trainval_idx, test_idx) in enumerate(skf.split(df, y)):

            print(f"\n\n=== StratifiedKFold FOLD {fold_id+1}/{n_splits} ===")

            # Split into trainval/test
            train_val_data = df.iloc[trainval_idx]
            test_data      = df.iloc[test_idx]

            # Optional internal train/val split

            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_size,
                shuffle=True,
                random_state=seed,
                stratify=train_val_data["Label_int"]
            )

            # Run one fold (your existing fold runner)
            row_summary = self._run_one_fold(
                fold_id=fold_id,
                train_df=train_data,
                val_df=val_data,
                test_df=test_data,
                mode="stratified_kfold",
            )

            self.cv_summaries.append(row_summary)

        return self.cv_summaries


    def _cv_groupkfold(self, df, n_splits=5, val_size=0.3, seed=0):

        """GroupKFold: keeps entire batches intact so train/test use disjoint batch_ids (leakage-safe across batches) but does not enforce class balance."""

        groups = df["batch_id"].to_numpy()
        gkf = GroupKFold(n_splits=n_splits)

        self.cv_summaries = []

        for fold_id, (trainval_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
            print(f"\n\n=== GroupKFold FOLD {fold_id+1}/{n_splits} ===")

            train_val_data = df.iloc[trainval_idx]
            test_data      = df.iloc[test_idx]

            # OPTIONAL: build val split inside train_val_data
            # If you want "no leakage" between train/val, use GroupShuffleSplit here too.
            gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
            tr_idx, va_idx = next(gss.split(train_val_data, groups=train_val_data["batch_id"]))
            train_data = train_val_data.iloc[tr_idx]
            val_data   = train_val_data.iloc[va_idx]

            row_summary = self._run_one_fold(
                fold_id=fold_id,
                train_df=train_data,
                val_df=val_data,
                test_df=test_data,
                mode="kfold",
            )
            self.cv_summaries.append(row_summary)

        return self.cv_summaries

    def _cv_leave_one_batch_out(self, df, val_size=0.3, seed=0):
        self.cv_summaries = []
        batches = df["batch_id"].unique()

        for fold_id, test_batch_id in enumerate(batches):
            print(f"\n\n=== LOBO FOLD {fold_id+1}/{len(batches)} | test_batch={test_batch_id} ===")

            train_val_data = df[df["batch_id"] != test_batch_id]
            test_data      = df[df["batch_id"] == test_batch_id]
            if self.include_rest:
                min_samples = train_val_data['Label_int'].value_counts().min()
                # downsample majority class
                idx_rest = train_val_data[train_val_data['Label_str']=='rest'].index.values
                index_rest_ds = train_val_data[train_val_data['Label_str']=='rest'].sample(n=min_samples, random_state=seed).index.values
                idx_to_drop = np.setdiff1d(idx_rest, index_rest_ds)
                train_val_data = train_val_data.drop(index=idx_to_drop)

            
            # same val split as before (NOT group-aware)
            train_data, val_data = train_test_split(train_val_data,test_size=val_size,shuffle=True,random_state=seed, stratify=train_val_data["Label_int"])

            row_summary = self._run_one_fold(
                fold_id=fold_id,
                train_df=train_data,
                val_df=val_data,
                test_df=test_data,
                mode="leave_one_batch_out",
                test_batch_id=int(test_batch_id),
            )
            self.cv_summaries.append(row_summary)

        return self.cv_summaries

    def _run_one_fold(self, fold_id, train_df, val_df, test_df, mode, test_batch_id=None):
        # fresh model master each fold
        self.model_master = Model_Master(self.base_config, self.model_config)
        self.model_master.df_train = train_df
        self.model_master.df_val   = val_df
        self.model_master.df_test  = test_df

        if self.model_master.kind == "ml":
            if self.model_config['model']['features']['scale_feats']:
                print("Scaling feats")
                feat_cols = self.model_master.extract_dataset_train_columns()
                scaler = StandardScaler()

                # fit on train only
                train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])

                # apply same transform to val/test
                val_df[feat_cols]   = scaler.transform(val_df[feat_cols])
                test_df[feat_cols]  = scaler.transform(test_df[feat_cols])

        self.model_master.generate_training_labels()
        self.model_master.remap_all_datasets()
        self.model_master.register_model()

        save_model_path = self.model_dire / f"{mode}_fold_{fold_id+1}"
        model, metrics, y_true, y_pred = self.model_master.train_model(test=True, save_model_path=save_model_path)

        row_summary = {
            "cv_mode": mode,
            "fold_num": int(fold_id + 1),
            "test_batch": int(test_batch_id) if test_batch_id is not None else None}
        # flatten metrics
        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                row_summary[k] = json.dumps(np.asarray(v).tolist())
            elif isinstance(v, (np.floating,)):
                row_summary[k] = float(v)
            else:
                row_summary[k] = v

        row_summary["train_idx"] = self.model_master.df_train.index.tolist()
        row_summary["val_idx"]   = self.model_master.df_val.index.tolist()
        row_summary["test_idx"]  = self.model_master.df_test.index.tolist()
        row_summary["y_true"]    = y_true.tolist()
        row_summary["y_pred"]    = y_pred.tolist()


        return row_summary

    def main(self):
        # Load data for the current subject
        df = pd.DataFrame()
        for curr_data_dire in self.data_dire_proc:
            df_curr = load_all_h5files_from_folder(curr_data_dire, key='wins_feats', print_statistics = False)
            df = pd.concat((df, df_curr), ignore_index=True)

        self.df = df.reset_index(drop=True)
        # Save statistics for the current experiment
        self.save_run_cfg()
        #print_dataset_summary_statistics()
        # Prepare CV splits (shared across all experiments)
        self.run_cv()
    
        ##################### POST TRAINING SAVINGS ###############################

        # Save summaries
        summary_df = pd.DataFrame(self.cv_summaries)

        summary_df.to_csv(self.model_dire / "cv_summary.csv")

        




if __name__ == "__main__":


    config_root = Path(__file__).resolve().parent.parent / "config"
    print("Config root:", config_root)

    base_config_path = config_root / "base_models_config.yaml"
    if not base_config_path.exists():
        print("Base config does not exist, return")
        sys.exit(1)
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------
    # Edit to add your model
    # ------------------------------------------------------------

    model_config_path = config_root / "models_configs" / "random_forest_config.yaml"
    #model_config_path = config_root / "models_configs" / "speechnet_base_with_padding.yaml"
    if not model_config_path.exists():
        print("Model Config does not exist, return")
        sys.exit(1)
    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------
    # Ablation settings (edit these lists)
    # ------------------------------------------------------------
    subjects = ["S01", "S02", "S03", "S04"]         # 
    #subjects =["S00"]
    conditions = ["silent", "vocalized"]            # or ["silent"] / ["vocalized"]

    # Safety: make sure original config doesn't already request pooled/all-subject training
    if isinstance(base_cfg["data"].get("subject_id"), list):
        print("WARNING: base_cfg.data.subject_id is a list (pooled/all-subject mode). "
              "This ablation loop expects per-subject runs. It will override it anyway.")

    # ------------------------------------------------------------
    # Run sweep
    # ------------------------------------------------------------
    for sub in subjects:
        for cond in conditions:
            print("\n" + "=" * 80)
            print(f"Running Global Model | subject={sub} | condition={cond}")
            print("=" * 80)

            # clone base config so each run is isolated
            cfg_run = deepcopy(base_cfg)
            cfg_run["data"]["subject_id"] = sub
            cfg_run["condition"] = cond

            # OPTIONAL: if you want deterministic but different seeds per setting:
            # cfg_run["experiment"]["seed"] = int(base_cfg["experiment"]["seed"]) + hash((sub, cond)) % 10000

            global_model_trainer = Global_Model_Trainer(base_config=cfg_run, model_config=model_cfg)
            global_model_trainer.main()


