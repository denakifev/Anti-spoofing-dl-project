# Voice Anti-Spoofing System

## Overview

This project implements a **Voice Anti-Spoofing** system designed to detect whether an audio sample is genuine (live human voice) or spoofed (e.g., replay attacks, synthesized or voice converted). The system enhances the security of speaker verification systems by preventing unauthorized access via voice-based attacks.

The model is based on the **Light CNN** architecture, trained and evaluated on the **La Partition of [ASVspoof 2019 Dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)**.

## Project Structure

- `train.py` — script to train the model
- `inference.py` — script to run inference on audio samples
- `src/` — source code including model, dataset handling, training, metrics, and utilities
- `requirements.txt` — Python dependencies

## Usage
By default, the dataset root directory is set via `datasets.la_root` and points to "/kaggle/input/asvpoof-2019-dataset/LA/LA/" — make sure to update this path to your local dataset location.

### Training
To start training run from directory root:

```bash
python train.py datasets.la_root="path/to/LA/root"
```
Also you can modify any parametrs of the model via hydra interface. See [Hydra intro docs](https://hydra.cc/docs/intro) and `configs/` directory.

### Inference

Make sure the model weights are downloaded and available locally in the runtime environment before running inference. If you use inference mode right after training, pretrained weights could be found in `saved/run/` and path to them is set by default as `inferencer.from_pretrained="saved/run/model_best.pth"`
To start inference mode run from directory root:

```bash
python inference.py inferencer.from_pretrained="path/to/your/pretrainded/weights" datasets.la_root="path/to/LA/root"
```
After the inference all saved data could be found in `dir_root/data/saved/`

## Metrics

### Equal Error Rate (ERR)

The main evaluation metric used is **Equal Error Rate (ERR)**, which represents the point where the false acceptance rate (FAR) and false rejection rate (FRR) are equal. A lower ERR indicates better anti-spoofing performance.

**Achieved result on eval dataset:**

```plaintext
ERR: 4.2283%
```

### Multiclass Accuracy
Additionally, **multiclass accuracy** was used to evaluate the model’s performance on the primary classes: **spoof** and **bonafide**.

**Achieved result on eval dataset:**

```plaintext
Multiclass Accuracy: 85%
```

## Loss Function

The model is trained using **Cross-Entropy Loss**, which is commonly used for classification tasks. This loss measures the difference between the predicted probability distribution and the true labels, encouraging the model to correctly classify genuine and spoofed audio samples.

## Model and Results
- **Light CNN** architecture implemented for anti-spoofing.

- Trained on the La Partition dataset.

- Detailed training and evaluation metrics available on [Weights & Biases](https://wandb.ai/den-akifev-hse-university/Voice_Anti_Spoofing_DL_Project/reports/Voise-Anti-Spoofing--VmlldzoxMzkxOTAzMA?accessToken=lu6hve0j781lrfg8gj4v1f2krbyv5jj1jnqhvjbbmthaiqvy9gftlcp7ejygbggu).

- Trained model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1VhPISuvvnRld86GsiZm0cgVEmIX2PQgS?usp=sharing).
