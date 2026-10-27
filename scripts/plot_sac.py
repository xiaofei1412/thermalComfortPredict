"""Plot the recorded first-chain reward curves without selecting checkpoints."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory, output):
    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    for seed, color in [(11, "#2563eb"), (23, "#d97706"), (37, "#059669")]:
        rows = [json.loads(s) for s in (directory/f"seed{seed}/learning_curve.jsonl").read_text(encoding="utf-8").splitlines()]
        ax.plot([r["actor_steps"] for r in rows], [r["reward_mean_last96"] for r in rows],
                color=color, label=f"Seed {seed}", linewidth=1.6)
    for index, day in enumerate([14, 35, 56, 63, 119, 147]):
        if index:
            ax.axvline(index*664, color="#cbd5e1", linewidth=.8, zorder=0)
        ax.text((index+.5)*664, 1.02, f"day {day}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=9, color="#475569")
    ax.axvline(664, color="#dc2626", linestyle="--", linewidth=1.2, label="Engineering restart")
    ax.set(xlim=(0, 3984), xlabel="Training actor transitions (prefix excluded)",
           ylabel="Mean reward (up to 96 recent steps)", title="First SAC chain: one configuration, all three seeds\n")
    ax.grid(axis="y", color="#e2e8f0", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=8, framealpha=.9)
    fig.supxlabel("Weather/episode changes every 664 steps; restart resets the window. These curves do not establish convergence.",
                  fontsize=8, color="#475569")
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.directory, args.output)
