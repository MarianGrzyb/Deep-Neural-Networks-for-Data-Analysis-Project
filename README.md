# Deep-Neural-Networks-for-Data-Analysis-Project

## Phase 2:

#### What’s there:
- [A] Setup of a project environment (hopefully with GPU support) [Kaggle Notebooks/Google Colab/your local machine] [1]
- [B] Coding the base model and training code [1]
- [C] Coding useful training infrastructure (logging results with e.g. Tensorboard, selecting and logging hyperparameters with e.g. Hparams (hyperparameter_tuning_with_hparams) or wandb.ai site or similar) [1, 2]
- [D] first run training (try to overfit to one train data batch – sanity check that training works), log progress & results dumped to easily searchable files (like plain txt) [1]
- [E] Demo application - where some test data can be inserted (live test data; alternatively some preexisting test data can be selected and loaded) and the model does its task and the result is presented back to the user [2]

#### How to get points (max. 15): upload project code and logs from the sanity run to eNauczenie site, pdf or presentation is not needed in this phase
- [C] Logging support [2 pts] [1]
- [C] Saving/Checkpointing support [2 pts] [2]
- [C] Hyperparameters selection logging [2 pts] [1]
- [B] Building the model (correct architecture and loss) [3 pts] [1]
- [D] Logs from a sanity check [2 pts] [1]
- [E] Demo application [4 pts] [2]

#### DEADLINE: 20.11.2026

#### Division
- Klaudia [1]
- Mateusz [2]


## Phase 2 - Klaudia [1] completed work

- [x] **[A]** Project environment: GitHub repository, Google Colab GPU runtime, Google Drive dataset storage, dependencies, and reproducible configuration.
- [x] **[B]** Dataset loader and preprocessing for the three provided classes: Benign, Malignant, and Normal.
- [x] **[B]** Pretrained ResNet-18 model with a three-class output layer.
- [x] **[B]** Weighted cross-entropy loss, AdamW optimiser, and training loop.
- [x] **[C]** TensorBoard logging, searchable JSONL/text logs, and hyperparameter logging.
- [x] **[D]** One-batch overfitting sanity check.

### Sanity-run result

The sanity check trained the model on one fixed training batch for 60 epochs on a Tesla T4 GPU.

| Metric | Result |
|---|---:|
| First weighted cross-entropy loss | 1.1656 |
| Final weighted cross-entropy loss | 0.0022 |
| Final training accuracy | 1.0000 |

The loss decreased strongly and accuracy reached 100%, confirming that the data pipeline, model, weighted loss, optimiser, backpropagation, and logging work correctly. This is a pipeline verification only, not a generalisation result.

### Reproducing the sanity run

```bash
python -m src.train --config configs/base.yaml --mode sanity --run-name sanity_v1
