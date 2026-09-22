# Thesis configurations

Every configuration needed to reproduce the results of *From Biosignals to Words*
(ETH Zurich, 2026). The folder follows the layout of the upstream `config/`:
windowing and base configurations at the top level, model configurations under
`models_configs/`.

A run always composes a **base** configuration, which pins the corpus, the window
length, the label set and the cross-validation scheme, with a **model**
configuration, which pins the architecture, the input domain, the objective and
the decoder:

```bash
python reproduce_paper_scripts/30_run_experiments.py \
  --base_config config_thesis/thesis_base_sentences_w2000_rest.yaml \
  --model_config config_thesis/models_configs/sentences_stft_ctc_transformer_classification_beam.yaml \
  --data_dir <DATA_DIR> --artifacts_dir artifacts_sentences \
  --experiment global --subjects S01 S02 S03 S04 S05 S06 S07 \
  --conditions silent vocalized
```

## Naming

Model configurations are named by the axes they fix, in a fixed order:

```
<corpus>_<domain>_<objective>_<sequence>[_<task>_<decoder>].yaml
```

| slot | values |
|---|---|
| `corpus` | `words` (15 words + rest, 1.4 s), `sentences` (20 sentences [+ rest], 2.0/2.4 s) |
| `domain` | `time`, `stft`, `mfcc_b15_q10` (EMG-fitted), `mfcc_b64_q40` (audio) |
| `objective` | `ce`, `ctc` |
| `sequence` | `none`, `bilstm`, `transformer` |
| `task` | `classification` (closed set against the lexicon), `recognition` (free character) |
| `decoder` | `greedy`, `beam`, `sweep` |

Cross-entropy has no decoding stage, so its files stop at the sequence slot. The
`sweep` files are trained with greedy decoding and dumped log-probabilities; the
beam grid is applied offline afterwards by `offline_experiments/VII_beam_sweep.py`,
so nothing is retrained. The two files named `speechnet_baseline_words_*` are the
published SpeechNet baseline, whose architecture and training recipe differ from
the thesis stack and are reproduced unchanged.

Base configurations are named `thesis_base_<corpus>_<window>_<label set>[_<normalization>]`.
Every field that decides which artefacts a run produces is pinned in the file
rather than left to a command-line flag, so a run cannot silently execute at the
wrong window length or with the wrong label set. A balanced accuracy over 21
classes averages in the recall of the rest class, which is easy and which is half
of the test windows, so figures obtained under different label sets are never
comparable.

| base configuration | window | label set | normalization |
|---|---|---|---|
| `thesis_base_words_w1400_rest.yaml` | 1.4 s | 15 words + rest | none |
| `thesis_base_sentences_w2000_rest.yaml` | 2.0 s | 20 sentences + rest | none |
| `thesis_base_sentences_w2000_norest.yaml` | 2.0 s | 20 sentences | none |
| `thesis_base_sentences_w2000_rest_zscore.yaml` | 2.0 s | 20 + rest | z-score, clip 2.6 σ |
| `thesis_base_sentences_w2000_rest_minmax.yaml` | 2.0 s | 20 + rest | percentile 2.5/97.5 |
| `thesis_base_sentences_onset_w2000_norest.yaml` | 2.0 s | 20 sentences | none, trigger-free windows |
| `thesis_base_sentences_onset_w2400_norest.yaml` | 2.4 s | 20 sentences | none, trigger-free windows |

The trigger-free extraction emits no rest window, so no `onset` base configuration
models rest: setting `include_rest: true` on that dataset trains a class that
never receives a sample instead of raising.

## Windowing

`create_windows_words.yaml` (1.4 s, word labels), `create_windows_sentences.yaml`
(2.0 s, sentence labels) and `create_windows_sentences_onset.yaml` (trigger-free,
anchored on the detected speech onset rather than on the cue) are read by
`reproduce_paper_scripts/20_make_windows_and_features.py`. The augmentation stride
and the number of shifts are declared there, so each point of the augmentation
ablation is one windowed dataset.

## Which configuration produces which result

| result | base | model |
|---|---|---|
| SpeechNet baseline, published corpus | `words_w1400_rest` | `speechnet_baseline_words_ce`, `speechnet_baseline_words_ctc` |
| SpeechNet baseline, corpus of this thesis | `words_w1400_rest` | `speechnet_baseline_words_ce` |
| input domains, words | `words_w1400_rest` | `words_<domain>_{ce_none,ce_bilstm,ctc_bilstm_classification_greedy}` |
| input domains, sentences | `sentences_w2000_rest` | `sentences_<domain>_{ce_none,ce_bilstm,ctc_bilstm_classification_greedy}` |
| sequence stages, closed set | `sentences_w2000_rest` | `sentences_stft_ctc_<arch>_classification_{greedy,beam}` |
| sequence stages, free character | `sentences_w2000_rest` | `sentences_stft_ctc_<arch>_recognition_{greedy,beam}` |
| decoder sweep | `sentences_w2000_rest` | `sentences_stft_ctc_<arch>_<task>_sweep` |
| EMG-fitted mel cepstrum | `sentences_w2000_rest` | `sentences_mfcc_b15_q10_ctc_<arch>_<task>_{greedy,beam}` |
| trigger-free windows | `sentences_onset_w2000_norest`, `sentences_onset_w2400_norest` | `sentences_stft_ctc_transformer_<task>_greedy` |
| cue-anchored reference for the trigger-free comparison | `sentences_w2000_norest` | `sentences_stft_ctc_transformer_<task>_greedy` |
| pooled multi-subject, normalization | `sentences_w2000_rest`, `..._zscore`, `..._minmax` | `sentences_stft_ctc_transformer_classification_greedy` |

`<arch>` is `bilstm` or `transformer`, `<domain>` is `time`, `stft` or
`mfcc_b64_q40`, `<task>` is `classification` or `recognition`.

## Two fields corrected against the runs

These configurations were checked field by field against the `run_cfg.json` written
beside every checkpoint of the 221 runs in the artefact tree. Two settings had
drifted in the working copies and are corrected here, so that the files reproduce
the reported numbers rather than a later edit:

* **Beam width and temperature.** Every STFT run decodes with prefix beam search at
  a width of ten and the three scoring terms at their defaults, which is what the
  thesis reports. The working copies had acquired the tuned mel values, a width of
  five and a temperature of 1.6. Those values are correct for the
  `mfcc_b15_q10` files, which keep them, and wrong for the STFT ones, which do not.
* **Optimizer.** Every run of the artefact tree trains with AdamW at `1e-3` and a
  plateau scheduler at factor 0.1 and patience 12. The published SpeechNet baseline
  keeps its own recipe, Adam at `1e-3` with patience 2, as reproduced from upstream.
