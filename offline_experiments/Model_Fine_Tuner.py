# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
File Containing Main Model Fine Tuner Class
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from offline_experiments.Model_Master import Model_Master
import torch


class Model_Fine_Tuner:
    def __init__(
        self,
        base_cfg: dict,
        model_cfg: dict,
        model_to_ft_path: Path,
        new_model_save_path: Path,
        ft_cfg_settings: dict,
        df_for_ft_train: pd.DataFrame,
        df_for_ft_val: pd.DataFrame,
    ):
        """
        Initializes the Model Fine-Tuner
        : base_cfg: Base Configuration (dict) containing general settings such as data paths, window size, etc.
        : model_cfg: Model Configuration (dict) containing settings specific to the model architecture, such
        : model_to_ft_path: Path to model to fine-tune
        : new_model_save_path : Path where to save the new model
        : ft_cfg_settings: Fine-Tuning Configuration (dict), containing settings such as whether to
                            fine-tune all layers or only specific layers, learning rate, number of epochs, etc.
        : df_for_ft_train: DataFrame Containing Data for Fine-Tuning
        : df_for_ft_val: DataFrame Containing Data for Validation during Fine-Tuning
        """

        self.model_cfg = model_cfg
        self.base_cfg = base_cfg

        self.model_to_ft_path = model_to_ft_path
        self.new_model_save_path = new_model_save_path
        self.ft_cfg_settings = ft_cfg_settings
        self.df_for_ft_train = df_for_ft_train
        self.df_for_ft_val = df_for_ft_val
        self.model_master = Model_Master(base_config=self.base_cfg, model_config=self.model_cfg)
        self.df_col = self.data_preparation_routine()
        self.model_master.register_model()
        print("Model registered!")
        self.model = self.build_and_load_weights()

    def build_and_load_weights(self):
        print("=====Loading weights!=======")

        # NEW: allow random init / from-scratch start
        if self.model_to_ft_path is None:
            print(
                "No checkpoint provided (model_to_ft_path=None) -> using RANDOMLY initialized weights."
            )
            return self.model_master.model

        checkpoint = torch.load(self.model_to_ft_path, map_location="cpu")
        state_dict = (
            checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        )

        # Load directly into the *existing* model object that the trainer references
        missing, unexpected = self.model_master.model.load_state_dict(state_dict, strict=False)

        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
        if len(missing) < 20 and missing:
            print("  missing:", missing)
        if len(unexpected) < 20 and unexpected:
            print("  unexpected:", unexpected)

        print("Model weights loaded successfully")
        return self.model_master.model

    def data_preparation_routine(self):
        """
        Data Preparation Routine for Fine-Tuning
        This function will prepare the data for fine-tuning, including generating training labels, remapping datasets, and extracting the relevant columns for training.
        """
        self.model_master.df_train = self.df_for_ft_train

        self.model_master.df_val = self.df_for_ft_val
        self.model_master.generate_training_labels()
        self.model_master.remap_all_datasets()
        df_col = self.model_master.extract_dataset_train_columns()
        df_col = df_col + ["Label_train"]
        return df_col

    def test_zero_shot_acc(self):
        print("Accuracy before starting fine tuning:")
        zero_shot_test_df = pd.concat((self.model_master.df_train, self.model_master.df_val))
        self.model_master.trainer_manager.test_loader = (
            self.model_master.trainer_manager.create_dataloader_from_df(
                zero_shot_test_df[self.df_col], batch_size=1, shuffle=False, num_workers=0
            )
        )
        metrics_before_ft, y_true, y_pred = self.model_master.trainer_manager.evaluate()
        return metrics_before_ft

    def main_ft(self):

        # Train the models
        # Check accuracy before starting the fine tuning process

        # For now - easy implementation - fine tune all layers
        self.model_master.trainer_manager.fit(
            save_model_path=self.new_model_save_path,
        )


if __name__ == "__main__":
    from models.seeds import *
