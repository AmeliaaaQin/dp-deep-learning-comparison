import matplotlib.pyplot as plt

# ======================
# Experimental Results
# ======================
noise = [0.5, 0.8, 1.0, 1.2, 1.5]
accuracy = [0.8667, 0.8645, 0.8668, 0.8741, 0.8745]
epsilon = [4.50, 0.60, 0.35, 0.26, 0.19]

# ======================
# Create Dual-Axis Plot
# ======================
fig, ax1 = plt.subplots(figsize=(7, 4.5))

# Left Y-axis: Accuracy
ax1.set_xlabel("Noise Multiplier")
ax1.set_ylabel("Test Accuracy", color="tab:blue")
ax1.plot(
    noise,
    accuracy,
    marker="o",
    linewidth=2,
    color="tab:blue",
    label="Test Accuracy"
)
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(0.86, 0.88)

# Right Y-axis: Privacy Budget (epsilon)
ax2 = ax1.twinx()
ax2.set_ylabel("Privacy Budget (ε)", color="tab:red")
ax2.plot(
    noise,
    epsilon,
    marker="s",
    linestyle="--",
    linewidth=2,
    color="tab:red",
    label="Privacy Budget (ε)"
)
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0, 5)

# ======================
# Title & Grid
# ======================
plt.title("Privacy–Utility Trade-off under DP-SGD")
ax1.grid(True, linestyle="--", alpha=0.6)

# ======================
# Legend (Combined)
# ======================
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="center right"
)

plt.tight_layout()
plt.show()


