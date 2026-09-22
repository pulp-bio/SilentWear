# One script per experiment unit

The scripts one level up run a whole axis of the decision chain in a single
process. These run one unit each, so that units may be assigned to separate
accelerators or restarted individually after a failure. They produce the same
artefacts under the same paths: running `30_axis3_bilstm_classification.sh` here
and running `../30_axis3_sequence_stage.sh` with `ARCHS=bilstm` write to the
same directory.

A unit is one model configuration over both protocols and both speaking
conditions. Restrict further with the environment variables of `../common.sh`,
which every script inherits:

```bash
# one unit
bash scripts/reproduce_thesis/single/30_axis3_none_classification.sh

# one protocol only
env EXPERIMENTS=global bash scripts/reproduce_thesis/single/30_axis3_none_classification.sh

# one accelerator per unit
env CUDA_VISIBLE_DEVICES=0 bash .../30_axis3_bilstm_classification.sh &
env CUDA_VISIBLE_DEVICES=1 bash .../30_axis3_bilstm_recognition.sh &
```

`env` is required in tcsh, which does not accept `VAR=value command`. In bash or
zsh the prefix may be written directly.

## The units

| prefix | axis | units |
|---|---|---|
| `10_axis1_*` | input domain | 12: two corpora × three domains × two sequence stages, cross-entropy, three participants |
| `20_axis2_*` | training objective | 2: the CTC counterpart of the STFT row, one per corpus |
| `30_axis3_*` | sequence stage | 10: three stages × two tasks, plus the mel cepstrum with two stages × two tasks |
| `40_axis4_*` | decoder | 4: two architectures × two tasks, each trained once and then swept offline |
| `50_axis5_*` | window anchor | 6: three window definitions × two tasks |
| `60_*` | not axes | 5: the two data ablations and the three pooled runs |

The numbering is that of the axis the unit belongs to, and the order matters:
each axis holds every earlier one at the value its predecessor selected, so a
unit of axis 3 assumes the STFT front-end and the CTC objective that axes 1
and 2 carried forward.
