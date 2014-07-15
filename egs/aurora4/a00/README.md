#My Aurora4 Setup.

## Basic clean GMM/HMM System

The training script for the basic clean training is in `00_01_train_clean.sh` and the brief summary of various systems' performance are in the table below. All the experiments will be under the folder `exp_clean`.

| Experiment | Average WER(%) | Description |
|:-----------|:--------------:|:------------|
| tri1a | 35.94 | 39D MFCC with per-utt CMVN |
| tri1b | 46.98 | 39D MFCC |
| tri1b | 18.02 | 39D MFCC + model based VTS |

## Basic multi-style System

The training script for the basic multi-style training is in `00_01_train_multi.sh` and the brief summary of various systems' performance are in the table below. All the experiments will be under the folder `exp_multi`.

#### GMM/HMM System

| Experiment | Average WER(%) | Description |
|:-----------|:--------------:|:------------|
| tri1a | 22.54 | 39D MFCC with per-utt CMVN |
| tri1b | 26.41 | 39D MFCC |
| tri1b | 19.68 | 39D MFCC + model based VTS |

#### DNN/HMM System

For all the DNN systems, we use 72D (24*3) FBank with per-utt MVN features.

##### 6 Hidden Layers
| Experiment | Average WER(%) | Description |
|:-----------|:--------------:|:------------|
| dnn1b | 13.75 | tri1a alignment |
| dnn1c | 13.66 | dnn1b alignment |
| *dnn1d* | *13.35* | dnn1c alignment |
| dnn1e | 13.51 | dnn1d alignment |

##### 7 Hidden Layers

| Experiment | Average WER(%) | Description |
|:-----------|:--------------:|:------------|
| dnn1d_7h | 13.41 | |
| *rbmdnn1a* | *13.09* | 1 RBM layer + 6 fine-tuned hidden layers |


