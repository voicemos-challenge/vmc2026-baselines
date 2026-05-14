# The baseline system of the VoiceMOS Challenge 2026 Track 3

Contact for questions about this baseline: Wen-Chin Huang (Nagoya University) wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp

This repository contains the baseline system for Track 3 of the VoiceMOS Challenge 2026.  
The task is to predict the speaker and accent similarity of a synthetic speech sample and a reference sample.

## Environment Setup

We use `uv` for environment management. We assume you are in a Linux environment. If you don't have `uv` installed, you can install it via `curl`.
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

All commands below start with `uv run` and `uv` should take care of everything as it reads `uv.lock`. Please do note that my machine has CUDA version 12.6, so if your machine has a different version, you might need to delete `uv.lock`, modify `pyproject.toml` and let `uv run` take care the rest.

## Dataset instructions

Please download the dataset from CodaBench, decompress it and put it anywhere you like. We will assume the path to it is `DATA_ROOT` from now on.  

```bash
tree $DATA_ROOT

$DATA_ROOT
├── README
├── sets
│   ├── dev.csv
│   └── train.csv
└── wav
    ├── vmc2026-track3-sys001-utt003.wav
    ...
```

For the training phase, we provide training set waveform samples with the corresponding similarity scores, as well as development set waveform samples without the scores. Participants can submit their development set prediction results to the CodaBench system to obtain the performance.

In particular, `DATA_ROOT/sets/train.csv` has the following header:

`system_id,utterance_id,listener_id,wav_a_path,wav_b_path,spk_sim,acc_sim`

Please note that these are _listener-wise_ scores, so there will be multiple rows with the same sample pair (but with different `listener_id`, `spk_sim` and `acc_sim`).

`DATA_ROOT/sets/dev.csv` has the following header:

`system_id,utterance_id,wav_a_path,wav_b_path`

## Trained models

For participants' reference, trained models and their corresponding inference result csv files are in `official-egs`.

```bash
$ tree official-egs/

official-egs/
├── acc_sim_adamw_lr1e-3
│   ├── acc_step20000_dev.csv
│   ├── model_acc_sim_step20000.pt
├── spk_sim_adamw_lr1e-3
│   ├── model_spk_sim_step20000.pt
│   ├── spk_step20000_dev.csv
└── zero_shot
    ├── acc_dev.csv
    └── spk_dev.csv
```

## Baseline 1: zero-shot cosine similarity using pre-trained `speechbrain/spkrec-ecapa-voxceleb`

The first baseline is to calculate the cosine similarity of the embeddings of the two samples using the pre-trained `speechbrain/spkrec-ecapa-voxceleb`, which . No training here, so this is a zero-shot setting.

### Inference

Run the following command to conduct inference and obtain the resulting csv:

```bash
uv run python inference.py --data-root <DATA_ROOT> --csv-path <DATA_ROOT>/sets/dev.csv --out egs/zero_shot/spk_dev.csv
```

More details of `speechbrain/spkrec-ecapa-voxceleb` can be found here: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

## Baseline 2: fine-tune `speechbrain/spkrec-ecapa-voxceleb` with a projection head.

The second baseline is to fine-tune `speechbrain/spkrec-ecapa-voxceleb` with the provided training set. A projection head takes the embeddings of the two samples as input and outputs the similarity score. Techniques like range clipping as described in https://arxiv.org/abs/2104.03017 and repetitive padding as described in https://arxiv.org/abs/2103.00110 were used. Training was conducted with a batch size of 16, the AdamW optimizer with learning rate 0.001, and a fixed number of training steps of 20,000.

### Fine-tuning

Say we want to fine-tune a model to predict speaker similarity. Run the following command to perform fine-tuning:

```bash
uv run python finetune.py --data-root <DATA_ROOT> --target-metric spk_sim --outdir egs/spk_sim
```

Use `--outdir` to specify where the model checkpoints will be saved. To fine-tune a model to predict accent similarity, simply pass `--target-metric acc_sim`.

### Inference

Using the trained model (say put in `egs/spk_sim/model_spk_sim_step20000.pt`), run the following command to conduct inference and obtain the resulting csv:

```bash
uv run python inference.py --data-root <DATA_ROOT> --csv-path <DATA_ROOT>/sets/dev.csv --checkpoint egs/spk_sim/model_spk_sim_step20000.pt --out egs/spk_sim/spk_step20000_dev.csv
```

## Baseline results

### Speaker similarity, dev set

|            | UTT-MSE | UTT-LCC | UTT-SRCC | SYS-MSE | SYS-LCC | SYS-SRCC |
|------------|---------|---------|----------|---------|---------|----------|
| Baseline 1 |  12.032 |   0.529 |    0.432 |  11.590 |   0.848 |    0.809 |
| Baseline 2 |   0.438 |   0.511 |    0.451 |   0.069 |   0.916 |    0.860 |

### Accent similarity, dev set

|            | UTT-MSE | UTT-LCC | UTT-SRCC | SYS-MSE | SYS-LCC | SYS-SRCC |
|------------|---------|---------|----------|---------|---------|----------|
| Baseline 1 |  11.997 |   0.448 |    0.369 |  11.606 |   0.809 |    0.749 |
| Baseline 2 |   0.418 |   0.465 |    0.440 |   0.060 |   0.902 |    0.861 |

## Acknowledgement and citation

This repo was a subset of [sheet](https://github.com/unilight/sheet), an open-source repo for speech quality assessment research. In addition, Gemini 3.1 Pro and ChatGPT 5.5 were used to assist the implementation of this repo.

Should you have any questions, please open an issue.

Should you use this baseline in your research, please cite the following:

```bibtex
@misc{vmc2026-track3-baseline,
  author = {Wen-Chin Huang},
  title = {Codebase for the baseline system of the VoiceMOS Challenge 2026 Track 3.},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/unilight/vmc2026-track3-baseline}}
}
```

Author: Wen-Chin Huang (Nagoya University)  
wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp