"""Plot the learning curve saved by mappo.py (one point per training update)."""
import json

import matplotlib.pyplot as plt
import numpy as np

returns = json.load(open("results/returns.json"))
window = max(1, min(25, len(returns) // 5))
smooth = np.convolve(returns, np.ones(window) / window, mode="valid")

plt.figure(figsize=(7, 4))
plt.plot(returns, alpha=0.3, label="mean team return per update")
plt.plot(range(window - 1, len(returns)), smooth, label=f"moving average ({window} updates)")
plt.xlabel("Training update (32 episodes each)")
plt.ylabel("Team return (higher = better)")
plt.title("MAPPO on MPE simple_spread (3 agents)")
plt.legend()
plt.tight_layout()
plt.savefig("results/learning_curve.png", dpi=150)
print("Saved results/learning_curve.png")
