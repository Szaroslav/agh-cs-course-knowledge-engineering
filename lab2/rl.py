import os
import math
import argparse
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

from rl.learner import CartPoleBalanceLearner, LunarLandingLearner
from rl.tuner import Tuner


def get_no_buckets_from_params(params: pd.Series) -> tuple[int, ...]:
    return tuple([
        int(params[index])
        for index in params.index if index.startswith("bucket_")
    ])


def get_base_filename(
    environment: Literal["cart-pole", "lunar-lander"],
    sarsa: bool = False
) -> str:
    return f"{environment.replace('-', '_')}_" \
        f"{'qlearning' if not sarsa else 'sarsa'}"


def main():
    parser = argparse.ArgumentParser(
        description="Q-Learning and SARSA — The Lunar Stick of the Truth"
    )
    parser.add_argument(
        "-v", "--environment",
        required=True,
        choices=["cart-pole", "lunar-lander"],
        help=
            "Environment ID used in evaluating or/and tuning. "
            "One of: 'cart-pole' and 'lunar-lander'."
    )
    parser.add_argument(
        "-s", "--sarsa",
        action="store_true",
        default=False,
        help="Use SARSA instead of Q-Learning"
    )
    parser.add_argument(
        "-e", "--evaluate",
        action="store_true",
        help="Evaluate the model"
    )
    parser.add_argument(
        "-t", "--tune",
        action="store_true",
        help="Tune hyperparameters of the model using Optuna"
    )
    parser.add_argument(
        "-d", "--drop",
        action="store_true",
        help="Drop study of the tuned hyperparameters using Optuna"
    )

    args = parser.parse_args()

    environment: Literal["cart-pole", "lunar-lander"] = args.environment
    sarsa: bool = args.sarsa
    drop_if_exists: bool = args.drop

    if args.tune:
        tune(environment, sarsa, drop_if_exists)
    if args.evaluate:
        evaluate(environment, sarsa)


def tune(
    environment: Literal["cart-pole", "lunar-lander"],
    sarsa: bool = False,
    drop_if_exists: bool = False,
) -> None:
    no_buckets = 4 if environment == "cart-pole" else 8

    tuner = Tuner(
        Learner=CartPoleBalanceLearner
            if environment == "cart-pole"
            else LunarLandingLearner,
        environment=environment,
        sarsa=sarsa,
        drop_if_exists=drop_if_exists,
        lr=(0.05, 0.5),
        lr_min=(0.001, 0.1),
        lr_decay=(0.9, 1.0),
        df=(0.9, 0.999),
        df_min=(0.85, 0.95),
        df_decay=(0.9, 1.0),
        er=(0.05, 0.5),
        er_min=(0.001, 0.1),
        er_decay=(0.9, 1.0),
        no_buckets=(
            tuple([2 for _ in range(no_buckets)]),
            tuple([7 for _ in range(no_buckets)]),
        ),
        no_jobs=-1,
        no_trials=1000 if environment == "cart-pole" else 5000,
        no_attempts_per_trial=500 if environment == "cart-pole" else 150
    )

    study = tuner.run()

    print("\nThe best parameters:", study.best_params)
    print("The best score:", study.best_value)

    results: DataFrame = study.trials_dataframe(
        attrs=("number", "value", "params")
    )
    results.rename(
        lambda c: (c
            .replace("number", "trial")
            .replace("value", "reward")
            .replace("params_", "")
        ),
        axis="columns",
        inplace=True
    )

    os.makedirs(os.path.join("results"), exist_ok=True)

    filename = f"{get_base_filename(environment, sarsa)}_tuning.csv"
    results.to_csv(
        os.path.join(
            "results",
            filename
        ),
        index=False
    )


def evaluate(
    environment: Literal["cart-pole", "lunar-lander"],
    sarsa: bool = False
) -> None:
    results = pd.read_csv(os.path.join(
        "results",
        f"{get_base_filename(environment, sarsa)}_tuning.csv"
    ))
    results.sort_values(
        by="reward",
        ascending=False,
        ignore_index=True,
        inplace=True
    )

    top_n = 5
    leading_zeros = math.ceil(math.log10(top_n))
    no_runs = 15
    no_attempts = 1000
    window = 10

    for i, params in results.head(top_n).iterrows():
        trial = int(params['trial'])

        print(f"\nModel {trial}: Reward = {params['reward']:.2f}")

        all_rewards: list[tuple[int, float]] = []

        for run in range(no_runs):
            Learner = (
                CartPoleBalanceLearner
                if environment == "cart-pole"
                else LunarLandingLearner
            )

            learner = Learner(
                sarsa=sarsa,
                lr=params["lr"],
                lr_min=params["lr_min"],
                lr_decay=params["lr_decay"],
                df=params["df"],
                df_min=params["df_min"],
                df_decay=params["df_decay"],
                er=params["er"],
                er_min=params["er_min"],
                er_decay=params["er_decay"],
                no_buckets=get_no_buckets_from_params(params),
            )

            rewards = learner.learn(max_attempts=no_attempts)
            all_rewards += [
                (attempt, rewards[attempt]) for attempt in range(len(rewards))
            ]

            print(f"Run {run + 1}: Last reward = {rewards[-1]:.2f}")

        rewards = pd.DataFrame(all_rewards, columns=["attempt", "reward"])
        rewards.insert(0, "run", np.repeat(np.arange(no_runs), no_attempts))
        rewards["reward_rolling"] = rewards.groupby("run")["reward"] \
            .transform(lambda r: r.rolling(window, min_periods=1).mean())

        sns.set_theme(style="darkgrid")

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=rewards,
            x="attempt",
            y="reward_rolling",
            label=f"Średnia krocząca o oknie {window} z {no_runs} uruchomień",
            errorbar=("sd", 0.5),
            err_kws={"label": "±½ oddchylenia standardowego"}
        )

        lr = params["lr"]
        lr_min = params["lr_min"]
        lr_decay = params["lr_decay"]
        df = params["df"]
        df_min = params["df_min"]
        df_decay = params["df_decay"]
        er = params["er"]
        er_min = params["er_min"]
        er_decay = params["er_decay"]

        no_buckets = get_no_buckets_from_params(params)

        plt.title(
            f"Uśrednione nagrody w {no_runs} uruchomieniach "
            "dla zestawu hiperparametrów:\n"
            f"α = {lr:.4f}, α_min = {lr_min:.4f}, α_decay = {lr_decay:.4f}\n"
            f"γ = {df:.4f}, γ_min = {df_min:.4f}, γ_decay = {df_decay:.4f}\n"
            f"ε = {er:.4f}, ε_min = {er_min:.4f}, ε_decay = {er_decay:.4f}\n"
            f"Rozmiar kubełków: ("
                f"{', '.join(map(lambda b: str(b), no_buckets))}"
            ")"
        )

        plt.xlabel("Krok")
        plt.ylabel("Nagroda")
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.join("results", "visuals"), exist_ok=True)

        filename = (
            f"{get_base_filename(environment, sarsa)}_{i:0{leading_zeros}}_"
            f"model_{trial}.png"
        )
        plt.savefig(os.path.join("results", "visuals", filename))

        plt.close()


if __name__ == '__main__':
    main()
