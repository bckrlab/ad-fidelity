#!/usr/bin/env bash

# train models and log to mlflow
python -m ad_fidelity.train --cn data/adni/CN --ad data/adni/AD --persistent-workers --lr 0.0001 --batch-size 64 --epochs 100 -o runs.json

# load models and compute attribution maps for test data
python -m ad_fidelity.attribute

# do fidelity test
python -m ad_fidelity.evaluate
