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
> Latent Demand Recovery implements multiple baselines, including TimesNet, ImputeFormer, SAITS, iTransformer, GPVAE, CSDI, and DLinear. The code is referenced from [PyPOTS](https://github.com/WenjieDu/PyPOTS/tree/main).
Links to the corresponding papers for each model are provided below:  
> - TimesNet: [*TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*](https://arxiv.org/abs/2210.02186)  
> - ImputeFormer: [*ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation*](https://dl.acm.org/doi/abs/10.1145/3637528.3671751)  
> - SAITS: [*SAITS: Self-attention-based imputation for time series*](https://www.sciencedirect.com/science/article/abs/pii/S0957417423001203)  
> - iTransformer: [*iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*](https://arxiv.org/abs/2310.06625)  
> - GPVAE: [*GP-VAE: Deep Probabilistic Time Series Imputation*](https://proceedings.mlr.press/v108/fortuin20a.html)  
> - CSDI: [*Conditional Sequential Deep Imputation for Irregularly-Sampled Time Series*](https://arxiv.org/abs/2010.02558)  
> - DLinear: [*Are Transformers Effective for Time Series Forecasting?*](https://ojs.aaai.org/index.php/AAAI/article/view/26317)  

```bash
cd latent_demand_recovery/exp
# Conduct MNAR evaluation on different models with various artificial missing rates, such as model=TimesNet and missing_rate=0.3
python app.py --model 'TimesNet' --missing_rate 0.3
# Perform demand recovery on raw data, reconstructing demand from censored sales
python app.py --model 'TimesNet'
```


### Forcasting
- SSA
>
```bash
cd demand_forecasting/SSA
# Perform demand forecasting on censored sales and recovered demand (which requires running Latent Demand Recovery first) using the similar scenario average method (statistics-based)
python ssa_forecasting.py
```

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
