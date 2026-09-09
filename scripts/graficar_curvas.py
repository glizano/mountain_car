"""Curvas de aprendizaje de los dos agentes, a partir de los CSV de historial."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def leer(p):
    with open(p, newline="") as f:
        return np.array([float(r["recompensa"]) for r in csv.DictReader(f)])

def media_movil(x, k):
    return np.convolve(x, np.ones(k) / k, mode="valid")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
cfg = [("qlearning", "Q-Learning tabular", 200, "#1f5c8b"),
       ("dqn", "DQN", 50, "#b3541e")]

for ax, (nombre, titulo, k, color) in zip(axes, cfg):
    r = leer(Path("resultados") / f"historial_{nombre}.csv")
    ax.plot(np.arange(len(r)), r, color=color, alpha=0.15, lw=0.5)
    suave = media_movil(r, k)
    ax.plot(np.arange(len(suave)) + k, suave, color=color, lw=1.8,
            label=f"media movil ({k} ep.)")
    ax.set_title(f"{titulo}  ({len(r)} episodios)", fontsize=11)
    ax.set_xlabel("episodio"); ax.set_ylabel("recompensa del episodio")
    ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("MountainCar-v0: curvas de aprendizaje", fontsize=12.5)
fig.tight_layout()
fig.savefig("resultados/curvas_aprendizaje.png", dpi=150)
print("guardado resultados/curvas_aprendizaje.png")

for nombre, titulo, k, _ in cfg:
    r = leer(Path("resultados") / f"historial_{nombre}.csv")
    print(f"  {titulo:<20} primeros 100: {r[:100].mean():8.2f} | "
          f"ultimos 100: {r[-100:].mean():8.2f} | mejor episodio: {r.max():.0f}")
