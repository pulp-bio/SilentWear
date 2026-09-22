# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Main Trainer for Classical ML-Model (scikit-learn)
"""

from models.utils import compute_metrics
from typing import Any, Optional, Tuple
import joblib
import pandas as pd
from pathlib import Path


class SklearnTrainer:
    def __init__(
        self,
        estimator: Any,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        label_col: str = "Label_train",
    ) -> None:
        self.model = estimator
        self.df_train = df_train
        self.df_test = df_test
        self.label_col = label_col

    def fit(self, save_model_path: Optional[Path] = None) -> Any:
        """
        Train sklearn model on features X and labels y.
        """

        # ----------------------------
        # Split train into X and y
        # ----------------------------
        X_train = self.df_train.drop(columns=[self.label_col])
        # print("Train set contains:", len(X_train.columns), "features")
        y_train = self.df_train[self.label_col]

        # ----------------------------
        # Fit estimator
        # ----------------------------
        self.model.fit(X_train, y_train)

        print("Sklearn model trained successfully.")

        if save_model_path is not None:
            model_name = f"{str(save_model_path.name)}.joblib"
            model_path = save_model_path.parent / Path(model_name)
            joblib.dump(self.model, model_path)
            print("Model saved at:", model_path)

        return self.model

    def evaluate(self) -> Tuple[dict, Any, Any]:
        """
        Evaluate model on test set.
        """
        X_test = self.df_test.drop(columns=[self.label_col])
        y_test = self.df_test[self.label_col]
        y_pred = self.model.predict(X_test)

        metrics, y_true, y_pred = compute_metrics(y_test, y_pred)

        return metrics, y_true, y_pred
