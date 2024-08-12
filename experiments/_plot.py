"""Plot fitness over generations for all experiments, averaged."""

import matplotlib.pyplot as plt
import pandas as pd
from .data_structures import Experiment, Generation, Individual, Population
import os
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from sqlalchemy import select
import sys


def main(file: str, metric: str, dbengine) -> None:
    """Run the program."""

    df = pd.read_sql(
        select(
            Experiment.id.label("experiment_id"),
            Generation.generation_index,
            Individual.fitness,
            Individual.novelty,
            Individual.age,
        )
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id),
        dbengine,
    )

    df["novelty"] = df["novelty"]
    df["age"] = (df["generation_index"] - df["age"]) / df["generation_index"]

    if "unique" in metric:
        agg_per_experiment_per_generation = (
            df.groupby(["experiment_id", "generation_index"])
            .agg({"novelty": pd.Series.nunique})
            .reset_index()
        )
        agg_per_experiment_per_generation.columns = [
            "experiment_id",
            "generation_index",
            "unique",
        ]

        agg_per_generation = (
            agg_per_experiment_per_generation.groupby("generation_index")
            .agg({"unique": ["mean", "std"]})
            .reset_index()
        )
        agg_per_generation.columns = [
            "generation_index",
            "unique_mean",
            "unique_std",
        ]

        plt.figure()

        # Plot max
        plt.plot(
            agg_per_generation["generation_index"],
            agg_per_generation[f"unique_mean"],
            label=f"Average Number of Unique Individuals",
            color="b",
        )
        plt.fill_between(
            agg_per_generation["generation_index"],
            [max(elem, 0) for elem in agg_per_generation["unique_mean"] - agg_per_generation["unique_std"]],
            agg_per_generation["unique_mean"] + agg_per_generation["unique_std"],
            color="b",
            alpha=0.2,
        )

        plt.xlabel("Generation index")
        plt.ylabel(metric)
        plt.title(f"Average Number  of Unique Individuals across repetitions with std as shade")
        plt.legend()
        plt.savefig(f"unique_{file.split('.')[0][-5:]}.png", dpi=200)


    else:
        agg_per_experiment_per_generation = (
            df.groupby(["experiment_id", "generation_index"])
            .agg({metric: ["max", "mean"]})
            .reset_index()
        )
        agg_per_experiment_per_generation.columns = [
            "experiment_id",
            "generation_index",
            "max",
            "mean",
        ]

        agg_per_generation = (
            agg_per_experiment_per_generation.groupby("generation_index")
            .agg({"max": ["mean", "std"], "mean": ["mean", "std"]})
            .reset_index()
        )
        agg_per_generation.columns = [
            "generation_index",
            "max_mean",
            "max_std",
            "mean_mean",
            "mean_std",
        ]

        plt.figure()

        # Plot max
        plt.plot(
            agg_per_generation["generation_index"],
            agg_per_generation[f"max_mean"],
            label=f"Max {metric}",
            color="b",
        )
        plt.fill_between(
            agg_per_generation["generation_index"],
            [max(elem, 0) for elem in agg_per_generation["max_mean"] - agg_per_generation["max_std"]],
            agg_per_generation["max_mean"] + agg_per_generation["max_std"],
            color="b",
            alpha=0.2,
        )

        # Plot mean
        plt.plot(
            agg_per_generation["generation_index"],
            agg_per_generation[f"mean_mean"],
            label=f"Mean {metric}",
            color="r",
        )
        plt.fill_between(
            agg_per_generation["generation_index"],
            [max(elem, 0) for elem in agg_per_generation["mean_mean"] - agg_per_generation["mean_std"]],
            agg_per_generation["mean_mean"]
            + agg_per_generation["mean_std"],
            color="r",
            alpha=0.2,
        )

        plt.xlabel("Generation index")
        plt.ylabel(metric)
        plt.title(f"Mean and max {metric} across repetitions with std as shade")
        plt.legend()
        plt.savefig(f"{os.path.dirname(__file__)}/plots/{metric}_{file.split('.')[0][-5:]}.png", dpi=200)

if __name__ == "__main__":
    file, *metrics = sys.argv[1:]

    setup_logging()
    dbengine = open_database_sqlite(
        file, open_method=OpenMethod.OPEN_IF_EXISTS
    )
    for metric in metrics:
        main(file, metric, dbengine)
