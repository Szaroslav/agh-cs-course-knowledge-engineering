import math

import gym


class QLearner:
    def __init__(self):
        self.environment = gym.make('CartPole-v1')
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]

    def learn(self, max_attempts):
        for _ in range(max_attempts):
            reward_sum = self.attempt()
            print(reward_sum)

    def attempt(self):
        observation = self.discretise(self.environment.reset())
        done = False
        reward_sum = 0.0
        while not done:
            self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def discretise(self, observation):
        return 1, 1, 1, 1

    def pick_action(self, observation):
        return self.environment.action_space.sample()

    def update_knowledge(self, action, observation, new_observation, reward):
        pass


def main():
    learner = QLearner()
    learner.learn(10000)


if __name__ == '__main__':
    main()
