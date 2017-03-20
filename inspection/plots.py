"""
Generate plots for the final paper.
"""

from matplotlib import pyplot as plt
plt.style.use("ggplot")

import numpy as np

# Should line plot
epoch_train_losses = np.array([
	6.1806202,
	4.7503452,
	3.8647115,
	3.3282607,
	3.0617297,
	2.9261239,
	2.8633306,
	2.8216517,
	2.8039637
])


# How to plot this shit
validation_loss = np.array([
	5.4378171,
	4.50532722,
	3.84883332,
	3.66190529,
	3.60085654,
    3.5258894,
    3.53464341,
    3.53921223,
    3.5437756
])

xs = np.arange(1, 10)
plt.plot(xs, epoch_train_losses)
plt.plot(xs, validation_loss)
plt.xlabel("Epoch")
plt.ylabel("Average Batch Cross Entropy")
plt.title("Loss Across Epochs")
plt.savefig("stats/loss.png")