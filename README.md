#### ⚠️ This is an official and supported fork of the [original](https://github.com/tinkoff-ai/exact) EXACT implementation. The original repository is on freeze and will not be update further.
# EXACT: How to Train Your Accuracy
The official implementation of the first multi-class accuracy relaxation loss.

🚀 The paper was published in [Pattern Recognition Letters](https://authors.elsevier.com/sd/article/S0167-8655(24)00203-4) (2024). 🚀

The short version of the paper was presented at the [TAG in Machine Learning ICML Workshop (2022)](https://icml.cc/virtual/2022/workshop/13447).

The preprint is available at [arXiv](https://arxiv.org/pdf/2205.09615.pdf).

## Loss implementation
The stand-alone implementation of the EXACT objective can be found in `exact/src/exact_pytorch/exact.py`.

## UCI Datasets

### Requirements
1. Install EXACT package: `pip install -e ./exact/`
2. Install UCI datasets package: `pip install -e ./uci-class/`

### Reproducing Results
In order to reproduce hyperparameter search run:
```
python uci-class/scripts/run.py <dataset name> --root <logging directory> -c hopt --method
<method>
```
To reproduce the final quality run:
```
python uci-class/scripts/run.py <dataset name> --root <logging directory> -c eval --method
<method> --lr <from hopt> --clip <from hopt> --margin <from hopt> --regularization <from hopt>
```

## Image Classification Datasets

### Requirements
1. Install MDN Metric package: `pip install -e ./mdn-metric`

### Reproducing Results
Generate the configs: `python ./mdn-metric/scripts/configs/generate-from-template.py ./mdn-metric/configs/exact/templates --best ./mdn-metric/configs/exact/best ./mdn-metric/configs/exact`

Hyperparameter search:
```
CUDA_VISIBLE_DEVICES=<gpu index> python -m mdn_metric hopt --config <path to config> --train-root
<training root> <path to dataset root>
```

Multi-seed evaluation:
```
CUDA_VISIBLE_DEVICES=<gpu index> python -m mdn_metric evaluate --config <path to config> --train-root
<training root> <path to dataset root>
```

Time and memory are measured with scripts in `./mdn-metric/scripts/performance/`.
