"""
File Containing Main Model Orchestrator
"""
import sys
from pathlib import Path

import torch
import pandas as pd
import torch.nn as nn

########### Project-level imports 
PROJECT_ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]     
sys.path.insert(0, str(PROJECT_ROOT))
from utils.I_data_preparation.experimental_config import ORIGINAL_LABELS, FS
from models.models_factory import ModelSpec, build_model_from_spec
from models.SklearnTrainer import * 
from models.TorchTrainer import * 
import re
from offline_experiments.general_utils import feature_names_to_consider, feature_columns_to_consider, reorder_ml_features_by_channel
#######################

class Model_Master:
    """
    Main Model Orchestrator:
    - builds label mappings based on include_rest
    - builds model from YAML model spec
    - Train the Model
    - Evaluates the Model
    """

    def __init__(self, base_config: dict, model_config: dict) -> None:
        self.base_config = base_config
        self.model_config = model_config
        # label maps
        self.original_label_map = ORIGINAL_LABELS.copy() 
        self.channel_order = self.base_config.get("channel_order", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15])
        #print(self.channel_order)
        #print(self.original_label_map)
        self.train_label_map = None            # train_id -> word
        self.train_to_orig = None              # train_id -> orig_id
        self.orig_to_train = None              # orig_id -> train_id (useful for dataset remap)
        self.num_classes = len(self.original_label_map)
        # Dataset Features or Channels to consider
        self.data_col_to_consider = None
        # model
        self.kind = self.model_config["model"]["kind"]
        self.model = None

        # Trainer
        self.trainer_manager = None

        # runtime
        self.device = torch.device(
            self.base_config.get("runtime", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Datasets
        self.df_train = pd.DataFrame()
        self.df_val = pd.DataFrame()
        self.df_test = pd.DataFrame()




    def generate_training_labels(self) -> None:
        """
        Generate:
          - train_label_map: {train_id: word}
          - train_to_orig:  {train_id: orig_id}
          - orig_to_train:  {orig_id: train_id}
          - num_classes
        """
        include_rest = self.base_config["experiment"]["include_rest"]
        original_map = self.original_label_map

        # Added distinction w.r.t. S00 (which had different labels)
        if self.base_config['data']['subject_id'] !='S00':
            if include_rest:
                print("Rest inclded")
                # identity mapping
                self.train_label_map = original_map.copy()
                self.train_to_orig = {k: k for k in original_map.keys()}
                self.orig_to_train = {k: k for k in original_map.keys()}
            else:
                # remove rest (assumes rest is orig label 0)
                filtered_items = [(k, v) for k, v in original_map.items() if k != 0]

                # train labels become 0..7
                self.train_label_map = {new_k: word for new_k, (_, word) in enumerate(filtered_items)}
                print(self.train_label_map)
                
                self.train_to_orig = {
                    new_k: orig_k for new_k, (orig_k, _) in enumerate(filtered_items)
                }
                self.orig_to_train = {
                    orig_k: new_k for new_k, (orig_k, _) in enumerate(filtered_items)
                }
        else:
            if include_rest == False: 
                filtered_items = [(k, v) for k, v in original_map.items() if k != 10]
            else:
                filtered_items = [(k, v) for k, v in original_map.items()]
            # train labels become 0..7
            self.train_label_map = {new_k: word for new_k, (_, word) in enumerate(filtered_items)}


            self.train_to_orig = {
                new_k: orig_k for new_k, (orig_k, _) in enumerate(filtered_items)
            }
            self.orig_to_train = {
                orig_k: new_k for new_k, (orig_k, _) in enumerate(filtered_items)
            }
        self.num_classes = len(self.train_label_map)
        print("Num classes set to:", self.num_classes)
        
    
    def apply_label_mapping(
            self, 
            df: pd.DataFrame,
            orig_to_train: dict[int, int],
            orig_col: str = "Label_int",
            out_col: str = "Label_train",
            drop_unmapped: bool = True,
        ) -> pd.DataFrame:
        """
        Map original labels (orig_col) -> training labels (out_col).
        Optionally drop rows whose orig label is not in orig_to_train.
        """
        
        df2 = df.copy()

        # map returns NaN for unmapped labels
        df2[out_col] = df2[orig_col].map(orig_to_train)

        if drop_unmapped:
            df2 = df2[df2[out_col].notna()].copy()

        # make it integer (after dropping NaNs)
        df2[out_col] = df2[out_col].astype(int)

        return df2

    def remap_all_datasets(self, label_col: str = "Label_int") -> None:

        """
        Applies orig_to_train mapping to train/val/test and prints summaries.
        Creates column 'Label_train'.
        """
        if self.orig_to_train is None or self.train_label_map is None:
            raise RuntimeError("Call generate_training_labels() before remapping datasets.")

        for name in ["df_train", "df_val", "df_test"]:
            df = getattr(self, name)
            if df.empty:
                print(f"Dataset {name} is empty, skipping")
                continue

            # print("mapping", name)
            # print(df.columns)
            df_mapped = self.apply_label_mapping(
                df,
                orig_to_train=self.orig_to_train,
                orig_col=label_col,
                out_col="Label_train",
                drop_unmapped=True,   # drops rest when include_rest=False
            )
            setattr(self, name, df_mapped)
            # ---- Print dataset statistics ----
            self.print_dataset_info(df_mapped, dataset_name=name)
    

    def extract_dataset_train_columns(self):
        """
        Function to return names of the columns contained in the training dataset
        This is needed since ML models use features, DL models use Directly EMG data
        
        :param self: 
        """
        if self.kind == 'ml':
            # Model config must specify features to consider
            features = feature_names_to_consider(
                        consider_time_feats=self.model_config.get("time_features", True),
                        consider_freq_feats=self.model_config.get("freq_features", True),
                        consider_wavelet_feats=self.model_config.get("wavelet_features", True),
                    )

            cols_train = feature_columns_to_consider(features, self.df_train)
            cols_test  = feature_columns_to_consider(features, self.df_test)

            # sanity: non-empty
            if not cols_train:
                raise ValueError("No feature columns found in df_train with the selected feature groups.")

            # ---- ASSERT SAME COLUMNS (exact match) ----
            set_train, set_test = set(cols_train), set(cols_test)
            if set_train != set_test:
                raise AssertionError("Feature columns don't match between train and test sets.\n")
            
            # ---- NEW APPLY CHANNEL ORDER ----
            cols_train = reorder_ml_features_by_channel(cols_train, self.channel_order)
            cols_test  = reorder_ml_features_by_channel(cols_test,  self.channel_order)

            
        elif self.kind == "dl":
            # decide which dataframe to inspect
            if not self.df_train.empty:
                df_ref = self.df_train
            elif not self.df_test.empty:
                df_ref = self.df_test
            else:
                raise ValueError("[Model_Master.py] Both training and Test Datasets are empty.")

            # extract all filtered channel columns
            ch_cols = df_ref.columns[
                df_ref.columns.str.contains(r"^Ch_") &
                df_ref.columns.str.contains(r"_filt$")
            ]
            ch_cols = list(ch_cols.values)

            # map channel index -> column name
            ch_dict = {}
            for col in ch_cols:
                m = re.search(r"Ch_(\d+)", col)
                if m:
                    ch_dict[int(m.group(1))] = col

            missing = [ch for ch in self.channel_order if ch not in ch_dict]
            if missing:
                raise ValueError(f"[Model_Master.py] Requested channels {missing} not found in dataset columns.")

            cols_train = [ch_dict[ch] for ch in self.channel_order]
        
        self.data_col_to_consider = cols_train
        #print("Training Columns will be: ")
        #print(self.data_col_to_consider)
        print("Total features:", len(self.data_col_to_consider))
        
        
        return cols_train
    

    def print_dataset_info(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Print dataset label statistics:

        - Included session and batch ids
        - distribution of Label_train
        - corresponding Label_str names
        - unique mapping Label_train -> Label_str
        """

        if df is None or len(df) == 0:
            print(f"{dataset_name}: EMPTY dataset")
            return

        print(f"\n== {dataset_name} ==")

        print(f"Contains data from sessions: {df['session_id'].unique()} - Batches: {df['batch_id'].unique()}")

        # ---- Distribution summary in one row ----
        counts = df["Label_train"].value_counts().sort_index()

        summary = ", ".join(
            [
                f"{i}({self.train_label_map.get(i, 'UNK')})={c}"
                for i, c in counts.items()
            ]
        )

        print(f"{dataset_name} label distribution: {summary}")

        # # ---- Unique mapping Label_train <-> Label_str ----
        # unique_pairs = (
        #     df[["Label_train", "Label_str"]]
        #     .drop_duplicates()
        #     .sort_values("Label_train")
        # )

        # print("\nUnique Label_train → Label_str mapping:")
        # for _, row in unique_pairs.iterrows():
        #     print(f"  {row['Label_train']} → {row['Label_str']}")


    def register_model(self) -> None:
        """
        Instantiate model based on ModelSpec + context.
        """
        model_name = self.model_config["model"]["name"]
        for suffix in ("_hparam_abl", "_abl", "_hparam_abl_lr"):
            if model_name.endswith(suffix):
                model_name = model_name[:-len(suffix)]
                break
        

        spec = ModelSpec(
            kind=self.model_config["model"]["kind"],
            name=model_name,
            kwargs=self.model_config["model"]["kwargs"], 
        )

        ctx = {
            "num_channels" : 14,                 # FIXED
            "num_samples" : int(self.base_config["window"]["window_size_s"] * FS), 
            "num_classes": self.num_classes,
        }

        self.model = build_model_from_spec(spec, ctx)

        # move to device only for DL models
        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
        
        # Compute dataset columns
        self.extract_dataset_train_columns()
        # Initialize also corresponding trainer class

        cols = self.data_col_to_consider + ["Label_train"]
        if self.kind == 'ml':
            
            self.trainer_manager = SklearnTrainer(estimator=self.model,
                                                df_train=self.df_train[cols], 
                                                df_val=None,
                                                df_test=self.df_test[cols], 
                                                label_col= 'Label_train')
        elif self.kind == 'dl':
            self.trainer_manager = TorchTrainer(estimator=self.model, 
                                                
                                                df_train=self.df_train[cols] if not self.df_train.empty else None, 
                                                df_val=self.df_val[cols] if not self.df_val.empty else None, 
                                                df_test=self.df_test[cols] if not self.df_test.empty else None, 
                                                train_cfg=self.model_config['model']["kwargs"]["train_cfg"], 
                                                label_col = 'Label_train')

        print("Model and Trainer Initialized!")


    def train_model(self, 
                    save_model_path: Optional[Path] = None, 
                    test: Optional[bool] = True):
        """
        Main Model Trainer
        """
        self.model = self.trainer_manager.fit(save_model_path)


        if test:
            metrics, y_true, y_pred = self.trainer_manager.evaluate()

        # return the model
        return self.model, metrics, y_true, y_pred



