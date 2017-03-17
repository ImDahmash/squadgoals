#!/usr/bin/env bash

python3 train.py --embed_dim=300 --embed_path=data/squad/glove.squad.300d.npy --epochs=10 --batch_size=30

