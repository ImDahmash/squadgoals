#!/usr/bin/env bash

if [[ $(uname -a) =~ "Darwin" ]]; then
    PYTHON=python
else
    PYTHON=python3.5
fi

$PYTHON train.py --embed_dim=300 --embed_path=data/squad/glove.squad.300d.npy --epochs=10 --batch_size=30 --valid "$@"

