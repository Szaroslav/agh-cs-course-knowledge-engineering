from math import radians
from abc import ABC
from collections import defaultdict
from typing import Any, Literal, SupportsFloat, cast

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym

DiscreteObservationSpace = tuple[int, ...]


class Learner(ABC):
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
        no_buckets: DiscreteObservationSpace,
        lower_bounds: list[float],
        upper_bounds: list[float],
        environment_id: Literal["CartPole-v1", "LunarLander-v3"],
        sarsa: bool = False,
        render_mode: Literal[
            "human",
            "rgb_array",
            "ansi",
            "rgb_array_list"
        ] | None = None
    ) -> None:
        self.environment = gym.make(environment_id, render_mode=render_mode)

        self.attempt_no = 1
        self.rewards: list[float] = []
        self.actions = [a for a in range(self.environment.action_space.n)]

        self.Q: defaultdict[
            tuple[DiscreteObservationSpace, int],
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
            np.linspace(l_bounds, u_bounds, num=n + 1)
            for l_bounds, u_bounds, n in zip(
                lower_bounds,
                upper_bounds,
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

            self.update_q(
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
    ) -> DiscreteObservationSpace:
        return tuple([
            np.digitize(observation[i], self.buckets[i])
            for i in range(len(observation))
        ])

    def pick_action(self, observation: NDArray[np.float32]) -> int:
        random_action = True if np.random.rand() < self.er else False

        if random_action:
            return self.sample()
        return self.get_best_action(observation)

    def update_q(
        self,
        observation: DiscreteObservationSpace,
        new_observation: DiscreteObservationSpace,
        reward: float,
        action: int,
        next_action: int | None = None,
    ) -> None:
        current_q = self.Q[observation, action]

        if self.sarsa and next_action is not None:
            next_q = self.Q[new_observation, next_action]
        else:
            next_q = self.get_best_q(new_observation)

        self.Q[observation, action] = (
            (1.0 - self.lr) * current_q + self.lr * (reward + self.df * next_q)
        )

    def get_best_q(self, observation) -> float:
        return max(
            [self.Q[(observation, a)] for a in self.actions]
        )

    def get_best_action(self, observation) -> int:
        return max(
            [(self.Q[(observation, a)], a) for a in self.actions]
        )[1]


class CartPoleBalanceLearner(Learner):
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
        no_buckets: DiscreteObservationSpace,
        sarsa: bool = False,
        render_mode: Literal[
            "human",
            "rgb_array",
            "ansi",
            "rgb_array_list"
        ] | None = None
    ) -> None:
        environment = gym.make("CartPole-v1")

        lower_bounds = [
            environment.observation_space.low[0],
            -0.5,
            environment.observation_space.low[2],
            -radians(50)
        ]
        upper_bounds = [
            environment.observation_space.high[0],
            0.5,
            environment.observation_space.high[2],
            radians(50)
        ]

        environment_id = "CartPole-v1"

        super().__init__(
            lr,
            lr_min,
            lr_decay,
            df,
            df_min,
            df_decay,
            er,
            er_min,
            er_decay,
            no_buckets,
            lower_bounds,
            upper_bounds,
            environment_id,
            sarsa,
            render_mode,
        )


class LunarLandingLearner(Learner):
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
        no_buckets: DiscreteObservationSpace,
        sarsa: bool = False,
        render_mode: Literal[
            "human",
            "rgb_array",
            "ansi",
            "rgb_array_list"
        ] | None = None
    ) -> None:
        lower_bounds = [-1.5, 0.0, -2.0, -2.0, -3.14, -5.0, 0.0, 0.0]
        upper_bounds = [ 1.5, 1.5,  2.0,  2.0,  3.14,  5.0, 1.0, 1.0]

        environment_id = "LunarLander-v3"

        super().__init__(
            lr,
            lr_min,
            lr_decay,
            df,
            df_min,
            df_decay,
            er,
            er_min,
            er_decay,
            no_buckets,
            lower_bounds,
            upper_bounds,
            environment_id,
            sarsa,
            render_mode,
        )
