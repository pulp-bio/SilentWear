# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
File Containing Main Model Orchestrator
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.I_data_preparation.experimental_config import FS, get_active_labels, build_label_maps
from models.models_factory import ModelSpec, build_model_from_spec
from models.utils import resolve_num_classes_from_cfg, save_model_architecture_to_csv
from models.SklearnTrainer import *
from models.TorchTrainer import *
from models.strategies import CrossEntropyStrategy, CTCStrategy, CTCRecognitionStrategy
from utils.I_data_preparation.ctc_text_mapper import CTCTextMapper, DEFAULT_BLANK_ID
import re
from offline_experiments.general_utils import (
    feature_names_to_consider,
    feature_columns_to_consider,
    reorder_ml_features_by_channel,
)


NUM_CHANNELS = 14


class Model_Master:
    """
    Main Model Orchestrator:
    - builds label mappings based on include_rest and label_mode
    - builds model from YAML model spec
    - Train the Model
    - Evaluates the Model
    """

    def __init__(self, base_config: dict, model_config: dict) -> None:
        self.base_config = base_config
        self.model_config = model_config
        # label maps
        self.label_mode = self.base_config.get("experiment", {}).get("label_mode", "word")
        self.original_label_map = get_active_labels(self.label_mode)
        self.channel_order = self.base_config.get(
            "channel_order", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15]
        )
        # print(self.channel_order)
        # print(self.original_label_map)
        self.train_label_map = None  # train_id -> word | sentence
        self.train_to_orig = None  # train_id -> orig_id
        self.orig_to_train = None  # orig_id -> train_id (useful for dataset remap)
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
            self.base_config.get("runtime", {}).get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        # Datasets
        self.df_train = pd.DataFrame()
        self.df_val = pd.DataFrame()
        self.df_test = pd.DataFrame()

    def generate_training_labels(self) -> None:
        """
        Generate:
          - train_label_map: {train_id: word | sentence}
          - train_to_orig:  {train_id: orig_id}
          - orig_to_train:  {orig_id: train_id}
          - num_classes
        """
        include_rest = bool(self.base_config["experiment"]["include_rest"])

        # S00 used a different label scheme and is handled separately.
        if self.base_config["data"]["subject_id"] != "S00":
            self.train_label_map, self.train_to_orig, self.orig_to_train = build_label_maps(
                self.label_mode, include_rest
            )
        else:
            original_map = self.original_label_map
            filtered_items = (
                [(k, v) for k, v in original_map.items() if k != 10]
                if not include_rest
                else list(original_map.items())
            )
            self.train_label_map = {new_k: text for new_k, (_, text) in enumerate(filtered_items)}

            self.train_to_orig = {new_k: orig_k for new_k, (orig_k, _) in enumerate(filtered_items)}
            self.orig_to_train = {orig_k: new_k for new_k, (orig_k, _) in enumerate(filtered_items)}
        self.num_classes = len(self.train_label_map)
        print(f"Label mode: {self.label_mode}")
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
            
            if df is None or df.empty:
                print(f"Dataset {name} is empty, skipping")
                continue

            # print("mapping", name)
            # print(df.columns)
            df_mapped = self.apply_label_mapping(
                df,
                orig_to_train=self.orig_to_train,
                orig_col=label_col,
                out_col="Label_train",
                drop_unmapped=True,  # drops rest when include_rest=False
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
        if self.kind == "ml":
            # Model config must specify features to consider
            features_cfg = self.model_config.get("model", {}).get("features", {}) or {}
            features = feature_names_to_consider(
                consider_time_feats=features_cfg.get("time_features", True),
                consider_freq_feats=features_cfg.get("freq_features", True),
                consider_wavelet_feats=features_cfg.get("wavelet_features", True),
            )

            cols_train = feature_columns_to_consider(features, self.df_train)
            cols_test = feature_columns_to_consider(features, self.df_test)

            # sanity: non-empty
            if not cols_train:
                raise ValueError(
                    "No feature columns found in df_train with the selected feature groups."
                )

            # ---- ASSERT SAME COLUMNS (exact match) ----
            set_train, set_test = set(cols_train), set(cols_test)
            if set_train != set_test:
                raise AssertionError("Feature columns don't match between train and test sets.\n")

            # ---- NEW APPLY CHANNEL ORDER ----
            cols_train = reorder_ml_features_by_channel(cols_train, self.channel_order)
            cols_test = reorder_ml_features_by_channel(cols_test, self.channel_order)

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
                df_ref.columns.str.contains(r"^Ch_") & df_ref.columns.str.contains(r"_filt$")
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
                raise ValueError(
                    f"[Model_Master.py] Requested channels {missing} not found in dataset columns."
                )

            cols_train = [ch_dict[ch] for ch in self.channel_order]

        else:
            raise ValueError(f"Unknown model kind: {self.kind}")

        self.data_col_to_consider = cols_train
        # print("Training Columns will be: ")
        # print(self.data_col_to_consider)
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

        print(
            f"Contains data from sessions: {df['session_id'].unique()} - Batches: {df['batch_id'].unique()}"
        )

        # ---- Distribution summary in one row ----
        counts = df["Label_train"].value_counts().sort_index()

        assert self.train_label_map is not None
        summary = ", ".join(
            [f"{i}({self.train_label_map.get(i, 'UNK')})={c}" for i, c in counts.items()]  # type: ignore[call-overload]
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

    @staticmethod
    def _get_loss_name(train_cfg: dict) -> str:
        """Validate and return the loss name from training config."""
        if "loss_name" not in train_cfg:
            raise KeyError(
                "Missing config key: model.kwargs.train_cfg.loss_name. "
                "Set it explicitly (e.g., 'ctc' or 'cross_entropy')."
            )
        loss_name = str(train_cfg["loss_name"]).lower().strip()
        valid_losses = {"ctc", "cross_entropy"}
        if loss_name not in valid_losses:
            raise ValueError(
                f"Unsupported loss_name='{loss_name}'. Supported values are: {sorted(valid_losses)}"
            )
        return loss_name

    def _build_ctc_mapper(self, train_cfg: dict) -> CTCTextMapper:
        """Build the CTC text mapper."""
        ctc_cfg = train_cfg.get("ctc")
        if not isinstance(ctc_cfg, dict):
            raise KeyError("For loss_name='ctc', model.kwargs.train_cfg.ctc must be provided.")

        lexicon_path = ctc_cfg.get("lexicon_path")
        if not lexicon_path:
            raise ValueError(
                "For loss_name='ctc', provide model.kwargs.train_cfg.ctc.lexicon_path."
            )

        decoding = str(ctc_cfg.get("decoding", "lexicon")).lower()
        return CTCTextMapper(
            lexicon_path=lexicon_path,
            train_label_map=self.train_label_map,
            blank_id=ctc_cfg.get("blank_id", DEFAULT_BLANK_ID),
            use_full_alphabet=(decoding == "recognition"),
        )

    def register_model(self) -> None:
        """
        Instantiate model based on ModelSpec + context.
        """
        model_name = self.model_config["model"]["name"]
        for suffix in ("_hparam_abl", "_abl", "_hparam_abl_lr"):
            if model_name.endswith(suffix):
                model_name = model_name[: -len(suffix)]
                break

        spec = ModelSpec(
            kind=self.model_config["model"]["kind"],
            name=model_name,
            kwargs=self.model_config["model"]["kwargs"],
        )

        ctx = {
            "num_channels": NUM_CHANNELS,
            "num_samples": int(self.base_config["window"]["window_size_s"] * FS),
            "num_classes": self.num_classes,
        }

        text_mapper = None
        train_cfg = None
        loss_name = None

        if self.kind == "dl":
            train_cfg = self.model_config["model"]["kwargs"]["train_cfg"]
            loss_name = self._get_loss_name(train_cfg)

            print(f"Selected loss function: {loss_name}")

            # CTC logits are character-level: output dimension must match token vocab size.
            if loss_name == "ctc":
                ctx["num_classes"] = resolve_num_classes_from_cfg(
                    self.base_config,
                    self.model_config,
                    train_label_map=self.train_label_map,
                )
                text_mapper = self._build_ctc_mapper(train_cfg)

        self.model = build_model_from_spec(spec, ctx)

        # Move to device only for DL models
        if isinstance(self.model, nn.Module):
            self.model.to(self.device)

        # Compute dataset columns
        self.extract_dataset_train_columns()
        # Initialize also corresponding trainer class
        assert self.data_col_to_consider is not None
        cols = self.data_col_to_consider + ["Label_train"]
        if self.kind == "ml":

            self.trainer_manager = SklearnTrainer(
                estimator=self.model,
                df_train=self.df_train[cols],
                df_test=self.df_test[cols],
                label_col="Label_train",
            )
        elif self.kind == "dl":
            assert train_cfg is not None
            if loss_name == "ctc":
                ctc_cfg = train_cfg["ctc"]
                decoding = str(ctc_cfg.get("decoding", "lexicon")).lower()
                decode_strategy = str(ctc_cfg.get("decode_strategy", "greedy")).lower()
                beam_width = int(ctc_cfg.get("beam_width", 10))
                beam_temperature = float(ctc_cfg.get("beam_temperature", 1.0))
                beam_blank_penalty = float(ctc_cfg.get("beam_blank_penalty", 0.0))
                beam_length_bonus = float(ctc_cfg.get("beam_length_bonus", 0.0))
                label_smoothing = float(ctc_cfg.get("label_smoothing", 0.0))
                if decoding == "recognition":
                    strategy = CTCRecognitionStrategy(
                        text_mapper,
                        decode_strategy=decode_strategy,
                        beam_width=beam_width,
                        beam_temperature=beam_temperature,
                        beam_blank_penalty=beam_blank_penalty,
                        beam_length_bonus=beam_length_bonus,
                        label_mode=self.label_mode,
                        label_smoothing=label_smoothing,
                    )
                else:
                    allow_nearest = ctc_cfg.get("allow_nearest", True)
                    # Classification decision rule: 'nearest' or 'score' 
                    # (exact closed-set ML via CTC scoring). Only meaningful in lexicon mode.
                    lexicon_decision = str(ctc_cfg.get("lexicon_decision", "nearest")).lower()
                    strategy = CTCStrategy(
                        text_mapper,
                        allow_nearest=bool(allow_nearest),
                        decode_strategy=decode_strategy,
                        beam_width=beam_width,
                        beam_temperature=beam_temperature,
                        beam_blank_penalty=beam_blank_penalty,
                        beam_length_bonus=beam_length_bonus,
                        label_smoothing=label_smoothing,
                        lexicon_decision=lexicon_decision,
                    )
            else:
                strategy = CrossEntropyStrategy()

            self.trainer_manager = TorchTrainer(
                estimator=self.model,
                df_train=self.df_train[cols] if not self.df_train.empty else None,
                df_val=self.df_val[cols] if not self.df_val.empty else None,
                df_test=self.df_test[cols] if not self.df_test.empty else None,
                train_cfg=train_cfg,
                label_col="Label_train",
                train_label_map=self.train_label_map,
                strategy=strategy,
            )

        print("Model and Trainer Initialized!")

        if self.kind == "dl":
            example_input = torch.zeros(
                1, ctx["num_channels"], ctx["num_samples"], device=self.device
            )
            save_model_architecture_to_csv(self.model, model_name, example_input=example_input)
            
    def train_model(self, save_model_path: Optional[Path] = None, test: bool = True):
        """
        Main Model Trainer
        """
        if self.trainer_manager is None:
            raise RuntimeError("Trainer not initialized. Call register_model() first.")

        self.model = self.trainer_manager.fit(save_model_path)

        if save_model_path is not None and self.base_config.get("plot_loss", False):
            model_path = save_model_path if save_model_path.suffix == ".pt" else save_model_path.with_suffix(".pt")
            if model_path.exists():
                try:
                    import torch
                    from utils.general_utils import plot_loss_curves
                    state = torch.load(model_path, map_location="cpu", weights_only=False)
                    if "train_loss" in state and "val_loss" in state:
                        plot_loss_curves(state["train_loss"], state["val_loss"], model_path)
                except Exception as e:
                    print(f"Failed to plot loss curves: {e}")

        if test:
            eval_kwargs = {}
            dump_path = self._ctc_logprob_dump_path(save_model_path)
            pred_txt_path = self._ctc_pred_txt_path(save_model_path)
            
            if dump_path is not None:
                eval_kwargs["dump_path"] = dump_path
            if pred_txt_path is not None:
                eval_kwargs["pred_txt_path"] = pred_txt_path
            
            metrics, y_true, y_pred = self.trainer_manager.evaluate(**eval_kwargs)
            
            return self.model, metrics, y_true, y_pred

        return self.model, None, None, None

    def _ctc_pred_txt_path(self, save_model_path: Optional[Path]) -> Optional[Path]:
        """Return the .txt path for the per-sample prediction dump, or None.

        Always on for DL + CTC runs but can be disabled with
        ``model.kwargs.train_cfg.ctc.dump_predictions: false``.
        """
        if self.kind != "dl" or save_model_path is None:
            return None
        train_cfg = self.model_config.get("model", {}).get("kwargs", {}).get("train_cfg", {})
        if str(train_cfg.get("loss_name", "")).lower().strip() != "ctc":
            return None
        ctc_cfg = train_cfg.get("ctc", {})
        if not bool(ctc_cfg.get("dump_predictions", True)):
            return None
        save_model_path = Path(save_model_path)
        return save_model_path.with_name(save_model_path.stem + "_predictions.txt")

    def _ctc_logprob_dump_path(self, save_model_path: Optional[Path]) -> Optional[Path]:
        """Return the .npz path to dump test log-probs to, or None if disabled.

        Enabled by ``model.kwargs.train_cfg.ctc.dump_logprobs: true`` in the model
        config (DL + CTC only).
        """
        if self.kind != "dl" or save_model_path is None:
            return None
        train_cfg = self.model_config.get("model", {}).get("kwargs", {}).get("train_cfg", {})
        if str(train_cfg.get("loss_name", "")).lower().strip() != "ctc":
            return None
        ctc_cfg = train_cfg.get("ctc", {})
        if not bool(ctc_cfg.get("dump_logprobs", False)):
            return None
        save_model_path = Path(save_model_path)
        return save_model_path.with_name(save_model_path.stem + "_logprobs.npz")