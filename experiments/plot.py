"""Графики и сводная таблица эксперимента с лимитом очереди и без него"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from experiments.backpressure import MODES, RESULTS


def load(mode: str) -> dict:
    """Прочитать данные одного прогона"""
    folder = RESULTS / mode
    return {
        "meta": json.loads((folder / "meta.json").read_text()),
        "queue": pd.read_csv(folder / "queue.csv"),
        "source": pd.read_csv(folder / "source.csv"),
        "containers": pd.read_csv(folder / "containers.csv"),
        "tasks": pd.read_csv(folder / "tasks.csv"),
    }


def second_worker_joined(data: dict) -> float:
    """Когда брокер увидел второго потребителя после запуска второго воркера

    Раньше смотреть нельзя, первый замер может показать воркер, оставшийся от прошлого прогона
    """
    queue = data["queue"]
    requested = data["meta"]["second_worker_requested_at"]
    joined = queue[(queue["consumers"] >= 2) & (queue["t"] >= requested)]
    return float(joined["t"].iloc[0]) if len(joined) else float("nan")


def summary(data: dict) -> dict:
    """Числа для таблицы в README"""
    queue, source, tasks = data["queue"], data["source"], data["tasks"]
    joined = second_worker_joined(data)
    progress = source[source["msg"] == "source progress"]
    finished = source[source["msg"] == "source finished"]
    containers = data["containers"]
    before = tasks[tasks["finished"] < joined]
    after = tasks[tasks["finished"] >= joined + 5]
    rates_before = progress[progress["t"] < joined]["rate"]
    rates_after = progress[progress["t"] >= joined + 5]["rate"]
    done = worker_rate(tasks)
    return {
        "max_depth": int(queue["ready"].max()),
        "max_queue_memory_mib": round(float(queue["queue_memory_mib"].max()), 2),
        "max_message_mib": round(float(queue["message_mib"].max()), 2),
        "max_broker_memory_mib": round(float(queue["broker_memory_mib"].max()), 1),
        "max_source_memory_mib": round(
            float(containers[containers["role"] == "source"]["mib"].max()), 1
        ),
        "source_seconds": round(float(finished["t"].iloc[0]), 1) if len(finished) else None,
        "source_rate_before": _median(rates_before),
        "source_rate_after": _median(rates_after),
        "worker_rate_before": _median(done[(done.index > 10) & (done.index < joined)]),
        "worker_rate_after": _median(
            done[(done.index >= joined + 5) & (done.index < done.index.max() - 5)]
        ),
        "latency_median_before": _median(before["latency"]),
        "latency_max_before": round(float(before["latency"].max()), 1),
        "latency_median_after": _median(after["latency"]),
        "latency_max": round(float(tasks["latency"].max()), 1),
        "rejected": int(finished["rejected"].iloc[0]) if len(finished) else None,
        "second_worker_joined": joined,
        "all_done_seconds": round(float(tasks["finished"].max()), 1),
        "attempts_max": int(tasks["attempts"].max()),
        "done": int((tasks["status"] == "done").sum()),
    }


def worker_rate(tasks: pd.DataFrame) -> pd.Series:
    """Сколько задач воркеры завершают в секунду, по времени окончания задач в базе"""
    seconds = tasks["finished"].dropna().astype(int)
    counts = seconds.value_counts().sort_index()
    counts = counts.reindex(range(counts.index.min(), counts.index.max() + 1), fill_value=0)
    return counts.rolling(3, center=True, min_periods=1).mean()


def _median(values: pd.Series) -> float | None:
    """Медиана, если есть значения"""
    values = values.dropna()
    return round(statistics.median(values), 1) if len(values) else None


def plot(runs: dict[str, dict], path: Path) -> None:
    """Четыре графика по каждому прогону, прогоны рядом"""
    figure, axes = plt.subplots(4, len(runs), figsize=(7 * len(runs), 14), squeeze=False)
    for column, (mode, data) in enumerate(runs.items()):
        queue, source, tasks = data["queue"], data["source"], data["tasks"]
        progress = source[source["msg"] == "source progress"]
        joined = second_worker_joined(data)
        depth, rate, latency, memory = axes[:, column]
        depth.set_title(MODES[mode], fontsize=14)

        depth.plot(queue["t"], queue["ready"], label="ждут в очереди")
        depth.plot(queue["t"], queue["unacked"], label="у воркеров")
        if data["meta"]["limit"]:
            depth.axhline(data["meta"]["limit"], color="grey", linestyle=":", label="лимит")
        depth.set_ylabel("задач")

        rate.plot(progress["t"], progress["rate"], marker=".", label="источник публикует")
        done = worker_rate(tasks)
        rate.plot(done.index, done.values, label="воркеры выполняют")
        rate.set_yscale("log")
        rate.set_ylabel("задач в секунду")

        latency.scatter(tasks["finished"], tasks["latency"], s=2)
        latency.set_ylabel("от приёма до результата, с")

        memory.plot(queue["t"], queue["queue_memory_mib"], label="процесс очереди")
        memory.plot(queue["t"], queue["message_mib"], label="тексты задач в очереди")
        memory.set_ylabel("память, МиБ")
        memory.set_xlabel("время от начала прогона, с")

        for axis in (depth, rate, latency, memory):
            axis.axvline(joined, color="red", linestyle="--", label="подключился второй воркер")
            axis.grid(alpha=0.3)
        for axis in (depth, rate, memory):
            axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    """Построить графики и записать сводку"""
    runs = {mode: load(mode) for mode in MODES if (RESULTS / mode / "meta.json").exists()}
    plot(runs, RESULTS / "backpressure.png")
    table = {mode: summary(data) for mode, data in runs.items()}
    (RESULTS / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
