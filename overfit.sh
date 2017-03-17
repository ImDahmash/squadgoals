#!/usr/bin/env bash

python3 train.py --embed_dim=300 --embed_path=data/squad/glove.squad.300d.npy --nosave --epochs=100 --batch_size=10 --subset=10

