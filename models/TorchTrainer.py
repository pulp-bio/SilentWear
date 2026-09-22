# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Main Trainer for Deep Learning Models (pytorch based)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from models.seeds import *
from models.utils import compute_metrics, compute_wer_metrics
from typing import Dict, List, Optional, Tuple, Literal, cast
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
import hashlib
from torch.optim.swa_utils import update_bn
from models.strategies import TaskStrategy, CTCStrategy, CTCRecognitionStrategy

DEFAULT_EARLY_STOP_PATIENCE = 5  # default value if not specified in train_cfg


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TorchTrainer:
    def __init__(self, estimator, df_train, df_val, df_test, train_cfg, label_col,
                 strategy: TaskStrategy, train_label_map: Optional[Dict[int, str]] = None):
        self.model = estimator
        self.df_train = df_train
        self.train_loader = None
        self.df_val = df_val
        self.valoader = None
        self.df_test = df_test
        self.test_loader = None
        self.train_cfg = train_cfg
        self.label_col = label_col
        self.train_label_map = train_label_map or {}

        if strategy is None:
            raise ValueError("A TaskStrategy (e.g., CrossEntropyStrategy or CTCStrategy) must be explicitly provided.")
        self.strategy = strategy

    def create_dataloader_from_df(
        self, df, batch_size=32, shuffle=False, num_workers=0  # we shuffle outside
    ):
        """
        Function to create a dataloader from a given df.
        """
        if df is None or df.empty:
            return None
        X_df = df.drop(columns=self.label_col)
        print(X_df.columns)

        # (N, T) per channel, then stack to (N, C, T)
        X_np = np.stack([np.stack(X_df[col].to_numpy()) for col in X_df.columns], axis=1).astype(
            np.float32
        )
        # X_np shape: (N, C, T)

        y_np = df[self.label_col].to_numpy()

        X_torch = torch.from_numpy(X_np)  # (N, C, T)
        print("Tensor data with shape", X_torch.shape)
        y_torch = torch.from_numpy(y_np)

        # Build dataset and dataloader
        dataset = TensorDataset(X_torch, y_torch)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

        return dataloader

    def build_scheduler(self, optimizer, scheduler_cfg, num_epochs):
        if not scheduler_cfg:
            return None

        name = scheduler_cfg["name"]
        print("Building scheduler:", name)
        if name in ("", "none", "null"):
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_cfg.get("T_max", num_epochs)),
                eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
            )
        elif name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.get("mode", "min"),
                factor=float(scheduler_cfg.get("factor", 0.1)),
                patience=int(scheduler_cfg.get("patience", 10)),
            )

        raise ValueError(f"Unknown scheduler: {name}")

    def train_loop(
        self,
        model,
        trainloader,
        valoader,
        train_cfg,
        save_path,
    ):

        if trainloader is None or valoader is None:
            raise ValueError(
                "train_loop requires non-empty train and validation dataloaders "
                "(got None). Check that df_train and df_val are not empty."
            )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Running on cuda")
            print(torch.cuda.device_count())
            print(torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("Running on CPU")

        model.to(device)

        # ---- Read config ----
        num_epochs = int(train_cfg.get("num_epochs", 50))

        # optimizer_cfg = train_cfg.get("optimizer", None) or {"name": "adam", "lr": 1e-3}
        optimizer_cfg = (
            train_cfg.get("optimizer_cfg", None)
            or train_cfg.get("optimizer", None)
            or {"name": "adam", "lr": 1e-3}
        )

        opt_name = str(optimizer_cfg.get("name", "adam")).lower()

        lr = float(optimizer_cfg["lr"])
        weight_decay = float(train_cfg.get("weight_decay", 0.0))
        # betas = optimizer_cfg.get("betas", (0.9, 0.999))

        print("Model will be trained for:", num_epochs, "epochs")
        early_stop_patience = train_cfg.get("early_stop_patience", DEFAULT_EARLY_STOP_PATIENCE)
        print("Early stop patienence set to:", early_stop_patience)
        print("Set optimizer", opt_name, "|lr:", lr, "|wd:", weight_decay)
        scheduler_cfg = train_cfg.get("scheduler", None)
        print("Set scheduler:", scheduler_cfg)
        grad_clip_norm = train_cfg.get("grad_clip_norm", None)
        if grad_clip_norm is not None:
            print("Set gradient clipping max_norm:", float(grad_clip_norm))

        # ----- Optimizer -----
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # ----- Scheduler (optional) -----
        scheduler = self.build_scheduler(optimizer, scheduler_cfg, num_epochs)

        # Warmup configuration: linearly scale LR during the first N epochs.
        # If `warmup_epochs` is set in `train_cfg`, we interpolate from 0->base_lr.
        warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        if warmup_epochs > 0:
            print(f"Warmup for {warmup_epochs} epochs; base_lrs: {base_lrs}")

        # -------- Zero-epoch (before training) evaluation --------
        model.eval()
        with torch.no_grad():
            # Train accuracy before training
            train_preds, train_tgts = [], []
            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device).long()
                out = model(x)
                preds = self.strategy.predict_labels(out, use_score_fallback=False)
                # ensure list-like for extend
                train_preds.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
                train_tgts.extend(y.cpu().numpy().tolist())
            train_acc_0 = np.mean(np.array(train_preds) == np.array(train_tgts))

            # Validation accuracy before training
            val_preds, val_tgts = [], []
            for x, y in valoader:
                x = x.to(device)
                y = y.to(device).long()
                out = model(x)
                preds = self.strategy.predict_labels(out, use_score_fallback=False)
                val_preds.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
                val_tgts.extend(y.cpu().numpy().tolist())
            val_acc_0 = np.mean(np.array(val_preds) == np.array(val_tgts))

        print(f"PRE-TRAIN | TRAIN ACC: {train_acc_0:.3f} | VAL ACC: {val_acc_0:.3f}")

        # -------- Real Training Starts --------
        # For the recognition path the reported quality metric is WER/CER. 
        # Val WER/CER are tracked at every epoch and the checkpoint is on the
        # primary metric (CER for word-mode, WER for sentence-mode), while the
        # scheduler and early-stopping follow the val loss.
        recog_strategy = (
            self.strategy if isinstance(self.strategy, CTCRecognitionStrategy) else None
        )
        monitor_name = (
            recog_strategy.primary_metric.upper() if recog_strategy is not None else "loss"
        )

        train_losses = []
        val_losses = []
        val_monitors = []          # per-epoch selection metric (WER/CER or loss)
        best_val_loss = float("inf")  # early-stopping signal (loss)
        best_monitor = float("inf")   # checkpoint-selection signal
        best_state = None

        patience = 0

        train_accs = []
        val_accs = []
        for epoch in range(num_epochs):
            # -------- Train --------
            # Warmup: scale learning rate linearly during initial epochs
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_scale = float(epoch + 1) / float(warmup_epochs)
                for i, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = base_lrs[i] * warmup_scale
            model.train()
            running_loss_train = 0.0
            train_batches = 0
            train_predictions = []
            train_targets = []

            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device).long()

                optimizer.zero_grad()
                outputs = model(x)
                loss = self.strategy.compute_loss(outputs, y, device)
                self.strategy.backward(loss)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()

                running_loss_train += loss.item()
                train_batches += 1
                preds = self.strategy.predict_labels(outputs, use_score_fallback=False)
                train_predictions.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
                train_targets.extend(y.cpu().numpy().tolist())
            train_accuracy = np.mean(np.array(train_predictions) == np.array(train_targets))
            avg_train_loss = running_loss_train / max(1, train_batches)
            train_accs.append(train_accuracy)

            # -------- Validation --------
            model.eval()
            running_loss_val = 0.0
            val_batches = 0
            val_predictions = []
            val_targets = []
            val_refs, val_hyps = [], []

            with torch.no_grad():
                for x, y in valoader:
                    x = x.to(device)
                    y = y.to(device).long()

                    outputs = model(x)
                    loss = self.strategy.compute_loss(outputs, y, device)

                    running_loss_val += loss.item()
                    val_batches += 1

                    if recog_strategy is not None:
                        val_hyps.extend(recog_strategy.predict_texts(outputs, force_greedy=True))
                        val_refs.extend(recog_strategy.reference_texts(y))
                    else:
                        preds = self.strategy.predict_labels(outputs)
                        val_predictions.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
                        val_targets.extend(y.cpu().numpy().tolist())

            avg_val_loss = running_loss_val / max(1, val_batches)

            # ----- Selection metric: WER/CER for recognition, else val loss -----
            if recog_strategy is not None:
                val_metrics, _, _ = compute_wer_metrics(val_refs, val_hyps, verbose=False, vocabulary=recog_strategy.word_vocabulary)
                monitor = float(val_metrics[recog_strategy.primary_metric])
                val_accuracy = float(np.mean([r == h for r, h in zip(val_refs, val_hyps)])) if val_refs else 0.0
                metric_str = (
                    f"VAL WER: {val_metrics['wer']:.3f} | VAL CER: {val_metrics['cer']:.3f} "
                    f"| SEL[{monitor_name}]: {monitor:.3f}"
                )
            else:
                monitor = float(avg_val_loss)
                val_accuracy = float(np.mean(np.array(val_predictions) == np.array(val_targets)))
                metric_str = f"TRAIN ACC: {train_accuracy:.3f} | VAL ACC: {val_accuracy:.3f}"
            val_accs.append(val_accuracy)
            val_monitors.append(monitor)

            # ----- Scheduler step (safe fallback) -----
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # print("stepping scheduler ReduceLROnPlateau")
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                print(f"{epoch} TRAIN loss: {avg_train_loss:.3f} | VAL loss: {avg_val_loss:.3f} | {metric_str} | LR: {current_lr:.2e}")
            else:
                print(f"{epoch} TRAIN loss: {avg_train_loss:.3f} | VAL loss: {avg_val_loss:.3f} | {metric_str}")
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Checkpoint selection on the monitor (primary WER/CER, or val loss).
            if monitor < best_monitor:
                best_monitor = monitor
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    # optional: store effective optimizer/scheduler settings used
                    "optimizer_cfg": optimizer_cfg,
                    "scheduler_cfg": scheduler_cfg,
                }

            # Early-stopping on the val loss, independent of selection.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = 0
            else:
                patience += 1
                print(f"Consecutive epochs without improvement: {patience}")
                if patience >= early_stop_patience:
                    print("Hit early stopping.")
                    break

        epochs_ran = len(train_losses)  # how many epochs actually executed
        best_epoch_1based = (best_state["epoch"] + 1) if best_state is not None else None

        # Save best model and loss history (UNCHANGED)
        if save_path is not None and best_state is not None:
            best_state.update(
                {
                    "train_loss": train_losses,
                    "val_loss": val_losses,
                    "train_acc": train_accs,
                    "val_acc": val_accs,
                    "val_monitor": val_monitors,
                    "monitor_name": monitor_name,
                    "best_monitor": float(best_monitor),
                    "best_val_loss": best_val_loss,
                    "requested_num_epochs": int(num_epochs),
                    "epochs_ran": int(epochs_ran),
                    "best_epoch": int(best_epoch_1based) if best_epoch_1based is not None else 0,
                    "early_stop_patience": int(early_stop_patience),
                }
            )
            torch.save(best_state, save_path)

        # Optionally restore best model weights before returning (UNCHANGED)
        if best_state is not None:
            model.load_state_dict(best_state["model_state_dict"])

        return model

    def fit(self, save_model_path: Optional[Path] = None):
        """
        Train Pytorch model on features X and labels y.
        """

        batch_size = int((self.train_cfg or {}).get("batch_size", 32))

        self.train_loader = self.create_dataloader_from_df(self.df_train, batch_size=batch_size)
        self.valoader = self.create_dataloader_from_df(self.df_val, batch_size=batch_size)
        self.test_loader = self.create_dataloader_from_df(self.df_test, batch_size=batch_size)
        # Check that splits are truly different
        self.check_data_splits()

        # Fit estimator
        if save_model_path is not None:
            save_model_path = Path(save_model_path)
            model_path = (
                save_model_path
                if save_model_path.suffix == ".pt"
                else save_model_path.with_suffix(".pt")
            )
        else:
            model_path = None

        self.model = self.train_loop(
            self.model, self.train_loader, self.valoader, self.train_cfg, model_path
        )
        return self.model

    def evaluate(self, test_loader=None, dump_path=None, pred_txt_path=None):
        """
        Evaluates the model on the test set and returns metrics, true labels, and predicted labels.
        If test_loader is provided, it will be used instead of the default test_loader created from df_test.
        If dump_path is provided (CTC strategies only), the per-sample test log-probs
        are also written to disk so decode parameters can be swept offline.
        If pred_txt_path is provided (CTC strategies only), a human-readable per-sample
        prediction file (pre/post lexicon constraint) is written.
        """
        loader = test_loader if test_loader is not None else self.test_loader
        return evaluate_model(
            self.model, loader, self.strategy, dump_path=dump_path, pred_txt_path=pred_txt_path
        )

    def check_data_splits(self):
        """
        Verify that train/val/test splits are disjoint.
        Detects overlap by index and by hashing sample content.
        Safe if one of the splits is None or empty.
        """
        # print("Checking data split integrity...")

        splits = {
            "TRAIN": getattr(self, "df_train", None),
            "VAL": getattr(self, "df_val", None),
            "TEST": getattr(self, "df_test", None),
        }

        # Keep only non-empty DataFrames
        valid = {}
        for name, df in splits.items():
            if df is None:
                # print(f" - {name}: None (skipping)")
                continue
            if df.empty:
                # print(f" - {name}: empty (skipping)")
                continue
            valid[name] = df
            # print(f" - {name}: {len(df)} samples")

        if len(valid) < 2:
            print("Not enough splits present to compare overlap (need at least 2).")
            print("Split integrity check complete.\n")
            return

        # ------------------------------------------------------------
        # 1) Check index overlap (fast)
        # ------------------------------------------------------------
        idx_sets = {name: set(df.index) for name, df in valid.items()}

        names = list(idx_sets.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                overlap = idx_sets[a].intersection(idx_sets[b])
                if overlap:
                    print(f"Overlap {a}–{b} (index): {len(overlap)} samples")
                else:
                    # print(f"No index overlap detected for {a}–{b}.")
                    continue

        # ------------------------------------------------------------
        # 2) Check overlap by sample content (robust)
        # ------------------------------------------------------------
        def hash_df(df):
            """
            Create a stable hash per row even if cells contain numpy arrays.
            """

            def row_fingerprint(row) -> str:
                h = hashlib.blake2b(digest_size=16)
                for v in row:
                    if isinstance(v, np.ndarray):
                        h.update(str(v.shape).encode())
                        h.update(str(v.dtype).encode())
                        h.update(v.tobytes())
                    elif isinstance(v, (list, tuple)):
                        arr = np.asarray(v)
                        h.update(str(arr.shape).encode())
                        h.update(str(arr.dtype).encode())
                        h.update(arr.tobytes())
                    else:
                        h.update(str(v).encode())
                return h.hexdigest()

            return df.apply(row_fingerprint, axis=1)

        hash_sets = {name: set(hash_df(df)) for name, df in valid.items()}

        names = list(hash_sets.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                overlap = hash_sets[a].intersection(hash_sets[b])
                if overlap:
                    print(f"Content overlap {a}–{b}: {len(overlap)} samples")
                else:
                    # print(f"No content overlap detected for {a}–{b}.")
                    continue

        print("Split integrity check complete.\n")


# ---------------------------------------------------------------------------
# CTC dumps and prediction files
# ---------------------------------------------------------------------------


def _dump_ctc_logprobs(
    dump_path,
    logprob_chunks: List[np.ndarray],
    label_chunks: List[np.ndarray],
    strategy: "CTCStrategy",
) -> None:
    """Persist test-set log-probs + reference labels for offline decode sweeps.

    Writes ``<dump_path>`` (an .npz with ``log_probs`` (N, T, C) fp16 and
    ``labels`` (N,) int64) plus a sibling ``<dump_path>.meta.json`` carrying the
    exact token<->char table and label<->text map, so VII_beam_sweep.py can rebuild the
    decoder offline without the data pipeline or the lexicon file.
    """
    if not logprob_chunks:
        return
    dump_path = Path(dump_path)
    dump_path = dump_path if dump_path.suffix == ".npz" else dump_path.with_suffix(".npz")
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    log_probs = np.concatenate(logprob_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0).astype(np.int64)

    mapper = strategy.text_mapper
    meta = {
        "task": "recognition" if isinstance(strategy, CTCRecognitionStrategy) else "classification",
        "blank_id": int(mapper.blank_id),
        "int_to_char": {str(int(k)): v for k, v in mapper.int_to_char.items()},
        "label_to_text_map": {str(int(k)): v for k, v in mapper.label_to_text_map.items()},
        "label_mode": getattr(strategy, "label_mode", None),
        "num_samples": int(labels.shape[0]),
        "num_frames": int(log_probs.shape[1]),
        "num_tokens": int(log_probs.shape[2]),
    }
    np.savez_compressed(dump_path, log_probs=log_probs, labels=labels)
    dump_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[DUMP] test log-probs -> {dump_path} ({log_probs.shape}, fp16)")


def _write_pred_txt(
    pred_txt_path,
    task: str,
    references: List[str],
    recognition_outputs: List[str],
    classification_outputs: List[str],
) -> None:
    """Write a per-sample prediction dump (``.txt`` + ``.csv``) for a CTC evaluation.

    The columns are task-specific -- a classification-only field is never shown for
    a recognition run (and vice-versa):
      - **recognition**: ``index, reference, recognition_output`` where
        ``reference`` is the ground-truth text and ``recognition_output`` is the
        free-character CTC decode (the hypothesis).
      - **classification**: the above plus ``classification_output`` -- the
        closed-set class text the sample was snapped to (the predicted class /
        sentence-or-word it converges to).

    The exact-match summary line only counts the fields that are actually present:
    a recognition dump reports recognition exact-match only. A ``.csv`` sibling with
    the same rows (header row, no ``#`` comments) is written for programmatic use.
    """
    import csv as _csv

    task = str(task).lower().strip()
    is_cls = task == "classification"

    pred_txt_path = Path(pred_txt_path)
    pred_txt_path = pred_txt_path if pred_txt_path.suffix == ".txt" else pred_txt_path.with_suffix(".txt")
    pred_txt_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(references)
    if is_cls:
        header = ["index", "reference", "recognition_output", "classification_output"]
        rows = [[i, r, h, c] for i, (r, h, c)
                in enumerate(zip(references, recognition_outputs, classification_outputs))]
    else:
        header = ["index", "reference", "recognition_output"]
        rows = [[i, r, h] for i, (r, h) in enumerate(zip(references, recognition_outputs))]

    correct_rec = sum(1 for r, h in zip(references, recognition_outputs) if r == h)
    summary = [f"recognition={correct_rec}/{n}"]
    if is_cls:
        correct_cls = sum(1 for r, c in zip(references, classification_outputs) if r == c)
        summary.append(f"classification={correct_cls}/{n}")

    comments = [
        f"# CTC prediction dump | task={task} | n_samples={n}",
        "# reference = ground-truth text | recognition_output = free-character CTC decode (hypothesis)",
    ]
    if is_cls:
        comments.append("# classification_output = closed-set class text after the lexicon constraint (predicted class)")
    comments.append(f"# exact-match: {'  '.join(summary)}")

    # .txt: comment header + tab-separated table (human-readable).
    txt_lines = comments + ["\t".join(header)]
    txt_lines += ["\t".join(str(v) for v in row) for row in rows]
    pred_txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    # .csv: same rows, plain header (machine-readable).
    csv_path = pred_txt_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[PRED] per-sample predictions ({task}) -> {pred_txt_path} (+ .csv)")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _label_ints_to_texts(mapper, label_ints) -> List[str]:
    """Map class label ids to their lexicon text, marking unmatched (-1) samples."""
    return [
        mapper.label_to_text_map.get(int(l), "<unmatched>") if int(l) >= 0 else "<unmatched>"
        for l in label_ints
    ]


def evaluate_model(
    model, test_loader, strategy: TaskStrategy, dump_path=None, pred_txt_path=None
) -> Tuple[Optional[dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute predictions and metrics on the test set using the provided model, test_loader, and strategy.

    If ``dump_path`` is set and the strategy is CTC-based, the per-sample test
    log-probs and reference labels are also written to disk (see
    :func:`_dump_ctc_logprobs`) for offline decode-parameter sweeps.

    If ``pred_txt_path`` is set and the strategy is CTC-based, a human-readable
    per-sample prediction file is written with the model output for both tasks
    (free-character decode before the lexicon constraint, and the class after it);
    see :func:`_write_pred_txt`.
    """
    if strategy is None:
        raise ValueError("A TaskStrategy must be explicitly provided for evaluation.")

    device = next(model.parameters()).device
    model.eval()

    ctc_strategy = strategy if isinstance(strategy, CTCStrategy) else None
    dump = dump_path is not None and ctc_strategy is not None
    want_txt = pred_txt_path is not None and ctc_strategy is not None
    logprob_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    # Free-character recognition path: decode to collapsed character text and
    # score with WER/CER (instead of accuracy/precision/recall/F1).
    if isinstance(strategy, CTCRecognitionStrategy):
        references, hypotheses = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                outputs = model(inputs)
                if dump:
                    logits = strategy._extract_logits(outputs)
                    logprob_chunks.append(F.log_softmax(logits, dim=-1).detach().cpu().to(torch.float16).numpy())
                    label_chunks.append(targets.detach().cpu().numpy())
                hypotheses.extend(strategy.predict_texts(outputs))
                references.extend(strategy.reference_texts(targets))
        if dump:
            _dump_ctc_logprobs(dump_path, logprob_chunks, label_chunks, strategy)
        if want_txt and references:
            mapper = strategy.text_mapper
            pred_ints, _ = mapper.texts_to_label_int(list(hypotheses), allow_nearest=True)
            cls_outputs = _label_ints_to_texts(mapper, pred_ints)
            _write_pred_txt(pred_txt_path, "recognition", references, hypotheses, cls_outputs)
        if not references:
            return None, None, None
        metrics, refs, hyps = compute_wer_metrics(references, hypotheses, vocabulary=strategy.word_vocabulary)
        return metrics, np.asarray(refs), np.asarray(hyps)

    all_targets = []
    all_preds = []
    raw_texts: List[str] = []      # free-character decode, before lexicon (for want_txt)
    ref_texts: List[str] = []      # reference texts (for want_txt)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            outputs = model(inputs)

            if dump or want_txt:
                logits = strategy._extract_logits(outputs)
                if dump and logits.ndim == 3:
                    logprob_chunks.append(F.log_softmax(logits, dim=-1).detach().cpu().to(torch.float16).numpy())
                    label_chunks.append(targets.detach().cpu().numpy())
                if want_txt and ctc_strategy is not None and logits.ndim == 3:
                    raw_texts.extend(ctc_strategy._decode_texts(logits))
                    ref_texts.extend(ctc_strategy.text_mapper.label_int_to_texts(targets))

            # Use strategy to obtain final predicted labels (handles CE and CTC)
            preds_np = strategy.predict_labels(outputs)

            # Normalize to numpy arrays
            if isinstance(preds_np, np.ndarray):
                batch_preds = preds_np
            else:
                try:
                    batch_preds = np.asarray(preds_np)
                except Exception:
                    # Fall back to converting torch tensors
                    batch_preds = preds_np.detach().cpu().numpy()

            all_preds.append(batch_preds)
            all_targets.append(targets.cpu().numpy())

    if dump and ctc_strategy is not None:
        _dump_ctc_logprobs(dump_path, logprob_chunks, label_chunks, ctc_strategy)

    # Concatenate batches along first axis
    if len(all_targets) == 0:
        return None, None, None

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    if want_txt and raw_texts and ctc_strategy is not None:
        # classification_output = predicted class text (AFTER the lexicon constraint);
        # recognition_output = raw free-character decode (BEFORE it).
        mapper = ctc_strategy.text_mapper
        cls_outputs = _label_ints_to_texts(mapper, y_pred.tolist())
        _write_pred_txt(pred_txt_path, "classification", ref_texts, raw_texts, cls_outputs)

    metrics, y_true, y_pred = compute_metrics(y_true, y_pred)

    return metrics, y_true, y_pred
