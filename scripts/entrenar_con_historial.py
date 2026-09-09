"""Entrena por tandas guardando el historial de recompensas en CSV.

Necesario porque en esta maquina los procesos en segundo plano mueren al
terminar cada comando, asi que el entrenamiento largo se hace en trozos y el
estado tiene que sobrevivir entre uno y otro.

Uso:  python scripts/entrenar_con_historial.py <qlearning|dqn> <episodios>
"""
import csv
import sys
from pathlib import Path

from mountain_car.agents import DQNAgent, QLearningAgent

AGENTES = {
    "qlearning": (QLearningAgent, Path("saves/qlearning_mountaincar.pkl")),
    "dqn": (DQNAgent, Path("saves/dqn_mountaincar.pt")),
}


def main() -> None:
    nombre, episodios = sys.argv[1], int(sys.argv[2])
    cls, ruta = AGENTES[nombre]
    csv_path = Path("resultados") / f"historial_{nombre}.csv"

    if ruta.exists():
        agente = cls.load(ruta)
    elif nombre == "dqn":
        # Hiperparametros ajustados: ver README, seccion de exploracion.
        agente = cls("MountainCar-v0", epsilon_decay=0.999,
                     explore_hold_min=15, explore_hold_max=40)
    else:
        agente = cls("MountainCar-v0")

    empieza_en = agente.training_episodes
    historial = agente.train(total_episodes=episodios, log_interval=max(episodios // 4, 1))
    agente.save(ruta)

    nuevo = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["episodio", "recompensa"])
        for i, r in enumerate(historial, start=empieza_en + 1):
            w.writerow([i, r])

    print(f"{nombre}: {len(historial)} episodios anexados a {csv_path} "
          f"(total acumulado {agente.training_episodes})")


if __name__ == "__main__":
    main()
