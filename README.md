#### Fork of the original implementation of the paper "EXACT: How to Train Your Accuracy" for the upcoming paper "EXACT for Free: A Robust Accuracy Optimization Method for Deep Learning and Beyond"
# EXACT for Free: A Robust Accuracy Optimization Method for Deep Learning and Beyond

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
