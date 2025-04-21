from typing import Literal

import numpy as np
import optuna
from optuna import Study, Trial
from optuna.trial import TrialState

from .learner import Learner as LearnerClass, DiscreteObservationSpace


class Tuner():
    def __init__(
        self,
        Learner: type[LearnerClass],
        environment: Literal["cart-pole", "lunar-lander"],
        lr: tuple[float, float],
        lr_min: tuple[float, float],
        lr_decay: tuple[float, float],
        df: tuple[float, float],
        df_min: tuple[float, float],
        df_decay: tuple[float, float],
        er: tuple[float, float],
        er_min: tuple[float, float],
        er_decay: tuple[float, float],
        no_buckets: tuple[DiscreteObservationSpace, DiscreteObservationSpace],
        sarsa: bool = False,
        drop_if_exists: bool = False,
        no_jobs: int = 1,
        no_trials: int = 100,
        no_attempts_per_trial: int = 500,
        no_validation_runs: int = 5,
    ) -> None:
        self.Learner = Learner

        self.environment = environment
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

        self.no_buckets = no_buckets

        self.drop_if_exists = drop_if_exists

        self.no_jobs = no_jobs
        self.no_trials = no_trials
        self.no_attempts_per_trial = no_attempts_per_trial
        self.no_validation_runs = no_validation_runs

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

        no_buckets = tuple([
            trial.suggest_int(
                f"bucket_{i}",
                self.no_buckets[0][i],
                self.no_buckets[1][i]
            )
            for i in range(len(self.no_buckets[0]))
        ])

        scores: list[float] = []

        for run in range(self.no_validation_runs):
            learner = self.Learner(
                sarsa=self.sarsa,
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
        study_name = f"{self.environment.replace("-", "_")}_" \
            + ("qlearning" if not self.sarsa else "sarsa")
        storage = "sqlite:///tuning.db"

        study_name_exists = study_name in optuna.get_all_study_names(storage)
        if self.drop_if_exists and study_name_exists:
            optuna.delete_study(study_name=study_name, storage=storage)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True
        )

        no_completed_trials = len(
            study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        )
        study.optimize(
            self.objective,
            n_trials=max(self.no_trials - no_completed_trials, 0),
            n_jobs=self.no_jobs
        )

        return study
