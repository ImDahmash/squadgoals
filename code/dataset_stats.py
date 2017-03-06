"""
Simple way to investigate and print statistics for the entire dataset.

This shows:
- Average length of a span
- Most common tokens occuring in spans
- Most common POS for beginning of span
"""

import sys
from collections import Counter
from pprint import pprint

from nltk import pos_tag

from tqdm import tqdm

def average_span_length():
    # Find average span length
    avg_span_length = 0.0
    count = 0

    with open("data/squad/train.span") as span_file:
        for line in span_file:
            start, end = line.split()
            start, end = float(start), float(end)
            avg_span_length += (end - start) + 1
            count += 1

    avg_span_length /= count
    return avg_span_length


def pos_tags():
    # Returns a histogram for POS tags starting an answer
    histogram_starts = Counter()
    histogram_ends = Counter()
    with open("data/squad/train.answer") as answer_file:
        for line in tqdm(answer_file.readlines(), desc="Tagging lines"):
            pos = [x[1] for x in pos_tag(line.split())]
            first_pos, last_pos = pos[0], pos[-1]
            histogram_starts[first_pos] += 1
            histogram_ends[last_pos] += 1
    return histogram_starts, histogram_ends

if __name__ == '__main__':

    # Ensure the POS Tagger model is present for nltk
    print("Ensuring tagger model present...")
    nltk.download("averaged_perceptron_tagger")

    print("Average length of spans: {:.2f} tokens".format(average_span_length()))

    first_pos_hist, last_pos_hist = pos_tags()
    print("POS First Token Histogram:")
    pprint(first_pos_hist)

    print("POS Last Token Histogram:")
    pprint(last_pos_hist)
