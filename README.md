# frn-50k-baseline

## Overview
The Repo is a baseline for Dataset [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), which accesses the complete pipeline used to train and evaluate.

You can discover the methodology and technical details behind FreshRetailNet-50K in [Technical Report](https://openreview.net/pdf?id=ObqFw6ah94)

## Running Experiments

### Requirements
It is recommended to create a new environment using conda.
```bash
conda create --name py3.8_frn python=3.8
conda activate py3.8_frn
pip install -r ./requirements.txt
```


### Latent Demand Recovery
[TODO]


### Forcasting

- SSA
[TODO]

- TFT
>Temporal Fusion Transformer (TFT) is a novel attention-based architecture which combines high-performance multi-horizon forecasting with interpretable insights into temporal dynamics.
>Paper link: https://arxiv.org/abs/1912.09363

To train and evaluate easily on raw data, run:
```bash
cd tft
python3 trainTFT.py    # train models
python3 predictTFT.py  # evaluate after finishing trainning
```

- DLinear
[TODO]
