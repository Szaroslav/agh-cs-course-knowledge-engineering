import os
import math
from collections import defaultdict
import argparse
from typing import Any, Literal, SupportsFloat, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
import gymnasium as gym
import optuna
from optuna import Study, Trial

import matplotlib.pyplot as plt
import seaborn as sns


class QLearner:
    def __init__(
        self,
        lr: float,
        lr_min: float,
        lr_decay: float,
        df: float,
        df_min: float,
        df_decay: float,
        er: float,
        er_min: float,
        er_decay: float,
        no_buckets: tuple[int, int, int, int],
        sarsa: bool = False,
        render_mode: Literal[
            "human",
            "rgb_array",
            "ansi",
            "rgb_array_list"
        ] | None = None
    ) -> None:
        self.environment = gym.make("CartPole-v1", render_mode=render_mode)

        self.attempt_no = 1
        self.rewards: list[float] = []
        self.actions = (0, 1)

        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]

        self.knowledge_base: defaultdict[
            tuple[tuple[int, int, int, int], int],
            float
        ] = defaultdict(float)

        self.sarsa = sarsa

        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay

        self.df = df
        self.df_min = df_min
        self.df_decay = df_decay

        self.er = er
        self.er_min = er_min
        self.er_decay = er_decay

        self.buckets = [
            np.linspace(l_bounds, u_bounds, num=n)
            for l_bounds, u_bounds, n in zip(
                self.lower_bounds,
                self.upper_bounds,
                no_buckets,
            )
        ]

        self.render_mode = render_mode

    def sample(self) -> int:
        return self.environment.action_space.sample().item()

    def learn(self, max_attempts: int) -> list[float]:
        for _ in range(max_attempts):
            self.attempt()

        return self.rewards

    def attempt(self):
        observation = self.discretise(self.environment.reset()[0])
        terminated, truncated = False, False
        reward_sum = 0.0

        while not truncated and not terminated:
            if self.render_mode:
                self.environment.render()

            action = self.pick_action(observation)

            new_observation, reward, terminated, truncated, _ = cast(
                tuple[
                    NDArray[np.float32],
                    SupportsFloat,
                    bool,
                    bool,
                    dict[str, Any],
                ],
                self.environment.step(action)
            )
            new_observation = self.discretise(new_observation)

            next_action = (
                self.pick_action(new_observation)
                if self.sarsa
                else None
            )

            self.update_knowledge(
                observation,
                new_observation,
                reward,
                action,
                next_action
            )

            observation = new_observation
            reward_sum += reward

        self.rewards.append(reward_sum)
        self.attempt_no += 1

        self.lr = max(self.lr * self.lr_decay, self.lr_min)
        self.df = max(self.df * self.df_decay, self.df_min)
        self.er = max(self.er * self.er_decay, self.er_min)

        return reward_sum

    def discretise(
        self,
        observation: NDArray[np.float32]
    ) -> tuple[int, int, int, int]:
        return tuple([
            np.digitize(observation[i], self.buckets[i])
            for i in range(len(observation))
        ])

    def pick_action(self, observation: NDArray[np.float32]) -> int:
        random_action = True if np.random.rand() < self.er else False

        if random_action:
            return self.sample()
        return self.get_best_action(observation)

    def update_knowledge(
        self,
        observation: tuple[int, int, int, int],
        new_observation: tuple[int, int, int, int],
        reward: float,
        action: int,
        next_action: int | None = None,
    ) -> None:
        current_q = self.knowledge_base[observation, action]

        if self.sarsa and next_action is not None:
            next_q = self.knowledge_base[new_observation, next_action]
        else:
            next_q = self.get_best_knowledge(new_observation)

        self.knowledge_base[observation, action] = (
            (1.0 - self.lr) * current_q
            + self.lr * (reward + self.df * next_q)
        )

    def get_best_knowledge(self, observation) -> float:
        return max(
            [self.knowledge_base[(observation, a)] for a in self.actions]
        )

    def get_best_action(self, observation) -> int:
        return max(
            [(self.knowledge_base[(observation, a)], a) for a in self.actions]
        )[1]


class Trainer():
    def __init__(
        self,
        Learner: type[QLearner],
        lr: tuple[float, float],
        lr_min: tuple[float, float],
        lr_decay: tuple[float, float],
        df: tuple[float, float],
        df_min: tuple[float, float],
        df_decay: tuple[float, float],
        er: tuple[float, float],
        er_min: tuple[float, float],
        er_decay: tuple[float, float],
        no_buckets: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
        no_trials: int = 100,
        no_attempts_per_trial: int = 500,
        no_validation_runs: int = 5,
    ) -> None:
        self.Learner = Learner

        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay

        self.df = df
        self.df_min = df_min
        self.df_decay = df_decay

        self.er = er
        self.er_min = er_min
        self.er_decay = er_decay

        self.no_buckets = no_buckets

        self.no_attempts_per_trial = no_attempts_per_trial
        self.no_validation_runs = no_validation_runs
        self.no_trials = no_trials

    def objective(self, trial: Trial) -> float:
        lr = trial.suggest_float("lr", self.lr[0], self.lr[1])
        lr_min = trial.suggest_float("lr_min", self.lr_min[0], self.lr_min[1])
        lr_decay = trial.suggest_float(
            "lr_decay",
            self.lr_decay[0],
            self.lr_decay[1]
        )

        df = trial.suggest_float("df", self.df[0], self.df[1])
        df_min = trial.suggest_float("df_min", self.df_min[0], self.df_min[1])
        df_decay = trial.suggest_float(
            "df_decay",
            self.df_decay[0],
            self.df_decay[1]
        )

        er = trial.suggest_float("er", self.er[0], self.er[1])
        er_min = trial.suggest_float("er_min", self.er_min[0], self.er_min[1])
        er_decay = trial.suggest_float(
            "er_decay",
            self.er_decay[0],
            self.er_decay[1]
        )

        bucket_x = trial.suggest_int(
            "bucket_x",
            self.no_buckets[0][0],
            self.no_buckets[1][0]
        )
        bucket_dx = trial.suggest_int(
            "bucket_dx",
            self.no_buckets[0][1],
            self.no_buckets[1][1]
        )
        bucket_theta = trial.suggest_int(
            "bucket_theta",
            self.no_buckets[0][2],
            self.no_buckets[1][2]
        )
        bucket_dtheta = trial.suggest_int(
            "bucket_dtheta",
            self.no_buckets[0][3],
            self.no_buckets[1][3]
        )
        no_buckets = (bucket_x, bucket_dx, bucket_theta, bucket_dtheta)

        scores: list[float] = []

        for run in range(self.no_validation_runs):
            learner = QLearner(
                lr=lr,
                lr_min=lr_min,
                lr_decay=lr_decay,
                df=df,
                df_min=df_min,
                df_decay=df_decay,
                er=er,
                er_min=er_min,
                er_decay=er_decay,
                no_buckets=no_buckets,
            )

            rewards = learner.learn(self.no_attempts_per_trial)

            scores.append(
                float(np.mean(rewards[-min(25, self.no_attempts_per_trial):]))
            )

        return float(np.mean(scores))

    def run(self) -> Study:
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, self.no_trials)

        return study


def main():
    parser = argparse.ArgumentParser(
        description="Q-Learning — The Stick of the Truth"
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
        default=True,
        help="Evaluate the model"
    )
    parser.add_argument(
        "-t", "--train",
        action="store_true",
        help="Train the model using Optuna"
    )

    args = parser.parse_args()

    if args.train:
        train(args.sarsa)
    if args.evaluate:
        evaluate(
            args.sarsa,
            params_csv_path=os.path.join(
                "results",
                "qlearning_training.csv"
            )
        )


def train(sarsa: bool = False) -> None:
    trainer = Trainer(
        Learner=QLearner,
        sarsa=sarsa,
        lr=(0.05, 0.5),
        lr_min=(0.001, 0.1),
        lr_decay=(0.9, 1.0),
        df=(0.9, 0.999),
        df_min=(0.85, 0.95),
        df_decay=(0.9, 1.0),
        er=(0.05, 0.5),
        er_min=(0.001, 0.1),
        er_decay=(0.9, 1.0),
        no_buckets=((2, 2, 2, 2), (8, 8, 8, 8)),
        no_trials=1000,
    )

    study = trainer.run()

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

    results.to_csv(
        os.path.join(
            "results",
            f"{"qlearning" if not sarsa else "sarsa"}_training.csv"
        ),
        index=False
    )


def evaluate(sarsa: bool = False, params_csv_path = str) -> None:
    results = pd.read_csv(params_csv_path)
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
            learner = QLearner(
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
                no_buckets=(
                    int(params["bucket_x"]),
                    int(params["bucket_dx"]),
                    int(params["bucket_theta"]),
                    int(params["bucket_dtheta"]),
                ),
            )

            rewards = learner.learn(max_attempts=no_attempts)
            all_rewards += [
                (attempt, rewards[attempt]) for attempt in range(len(rewards))
            ]

            print(f"Run {run + 1}: Last reward = {rewards[-1]:.2f}")

        df = pd.DataFrame(all_rewards, columns=["attempt", "reward"])
        df.insert(0, "run", np.repeat(np.arange(no_runs), no_attempts))
        df["reward_rolling"] = df.groupby("run")["reward"] \
            .transform(lambda r: r.rolling(window, min_periods=1).mean())

        sns.set_theme(style="darkgrid")

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
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

        bucket_x = int(params["bucket_x"])
        bucket_dx = int(params["bucket_dx"])
        bucket_theta = int(params["bucket_theta"])
        bucket_dtheta = int(params["bucket_dtheta"])

        plt.title(
            f"Uśrednione nagrody w {no_runs} uruchomieniach "
            "dla zestawu hiperparametrów:\n"
            f"α = {lr:.4f}, α_min = {lr_min:.4f}, α_decay = {lr_decay:.4f}\n"
            f"γ = {df:.4f}, γ_min = {df_min:.4f}, γ_decay = {df_decay:.4f}\n"
            f"ε = {er:.4f}, ε_min = {er_min:.4f}, ε_decay = {er_decay:.4f}\n"
            f"Rozmiar kubełków: ("
                f"{bucket_x}, {bucket_dx}, {bucket_theta}, {bucket_dtheta}"
            ")"
        )

        plt.xlabel("Krok")
        plt.ylabel("Nagroda")
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.join("results", "visuals"), exist_ok=True)

        filename = (
            ("qlearning" if not sarsa else "sarsa")
            + "_"
            + f"{i:0{leading_zeros}}"
            + "_model_"
            + str(trial)
            + ".png"
        )
        plt.savefig(os.path.join("results", "visuals", filename))

        plt.close()


if __name__ == '__main__':
    main()
