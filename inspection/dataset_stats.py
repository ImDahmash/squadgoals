"""
Simple way to investigate and print statistics for the entire dataset.

This shows:
- Average length of a span
- Most common tokens occuring in spans
- Most common POS for beginning of span
"""

from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt
import nltk
import numpy as np
from tqdm import tqdm

import os

def average_span_length():
    # Find average span length
    distribution = []

    avg_span_length = 0.0
    count = 0

    with open("data/squad/train.span") as span_file:
        for line in tqdm(span_file.readlines(), desc="Calculating span length"):
            start, end = line.split()
            start, end = float(start), float(end)
            length = end - start + 1
            distribution.append(length)
            avg_span_length += length
            count += 1

    avg_span_length /= count
    return avg_span_length, distribution


def pos_tags():
    histogram_starts = Counter()
    histogram_ends = Counter()

    with open("data/squad/train.answer") as answer_file:
        for line in tqdm(answer_file.readlines(), desc="Tagging lines"):
            pos = [x[1] for x in nltk.pos_tag(line.split())]
            first_pos, last_pos = pos[0], pos[-1]
            histogram_starts[first_pos] += 1
            histogram_ends[last_pos] += 1

    return histogram_starts, histogram_ends


def plot_bar_from_counter(counter, n=10):
    frequencies = [x[1] for x in counter.most_common(n)]
    names = [x[0] for x in counter.most_common(n)]

    plt.figure()
    ax = plt.subplot(111)

    x_coordinates = np.arange(len(names))
    ax.bar(x_coordinates, frequencies, align='center')

    ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(names))

    return ax

if __name__ == '__main__':

    # Ensure directories are formed
    os.makedirs("stats", exist_ok=True)

    # Ensure the POS Tagger model is present for nltk
    print("Ensuring tagger model present...")
    nltk.download("averaged_perceptron_tagger")

    avg_span, spans_distribution = average_span_length()
    print("Average length of spans: {:.2f} tokens".format(avg_span))

    plt.hist(spans_distribution, bins=np.arange(0, 10, 1))
    plt.xlabel("Span Length")
    plt.ylabel("Frequency")
    plt.savefig("stats/spans.png")

    first_pos_hist, last_pos_hist = pos_tags()
    print("POS First Token Histogram:")
    pprint(first_pos_hist)

    plot_bar_from_counter(first_pos_hist)
    plt.xlabel("First Token POS Tag")
    plt.ylabel("Frequency")
    plt.savefig("stats/pos_first.png")

    print("POS Last Token Histogram:")
    pprint(last_pos_hist)

    plot_bar_from_counter(last_pos_hist)
    plt.xlabel("Last Token POS Tag")
    plt.ylabel("Frequency")
    plt.savefig("stats/pos_last.png")
