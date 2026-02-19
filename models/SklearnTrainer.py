"""
Main Trainer for Classical ML-Model (scikit-learn)
"""

from models.utils import compute_metrics
from typing import Optional
import joblib 
from pathlib import Path

class SklearnTrainer:
    def __init__(self, 
                estimator, 
                df_train, 
                df_val, 
                df_test, 
                label_col = 'Label_train'):
        self.model = estimator

        self.df_train = df_train
        self.df_val = None              # Not needed for Sklearn 
        self.df_test = df_test
        self.label_col = label_col

    def fit(self, 
            save_model_path : Optional[Path] = None):
        """
        Train sklearn model on features X and labels y.
        """

        # ----------------------------
        # Split train into X and y
        # ----------------------------
        X_train = self.df_train.drop(columns=[self.label_col])
        #print("Train set contains:", len(X_train.columns), "features")
        y_train = self.df_train[self.label_col]

        # ----------------------------
        # Fit estimator
        # ----------------------------
        self.model.fit(X_train, y_train)


        print("Sklearn model trained successfully.")

        if save_model_path is not None:
            model_name = f"{str(save_model_path.name)}.joblib"
            model_path  = save_model_path.parent / Path(model_name)
            joblib.dump(self.model, model_path)
            print("Model saved at:", model_path)


        return self.model

    def evaluate(self):
        """
        Evaluate model on test set.
        """
        X_test = self.df_test.drop(columns=[self.label_col])
        y_test = self.df_test[self.label_col]
        y_pred = self.model.predict(X_test)

        metrics, y_true, y_pred = compute_metrics(y_test, y_pred)

        return metrics, y_true, y_pred

