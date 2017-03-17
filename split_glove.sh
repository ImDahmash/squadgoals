#!/usr/bin/env bash

DIM=100 # Change to modify

GLOVE_PATH="data/dwr/glove.840B.300d.txt"
OUTPUT_WORDS="data/dwr/token.300d.txt"
OUTPUT_VECS="data/dwr/vec.300d.txt"

echo "Writing tokens to $OUTPUT_WORDS"
cut -f1 -d' ' $GLOVE_PATH > $OUTPUT_WORDS

echo "Writing vectors to $OUTPUT_VECS"
cut -f2-1000 -d' ' $GLOVE_PATH > $OUTPUT_VECS
