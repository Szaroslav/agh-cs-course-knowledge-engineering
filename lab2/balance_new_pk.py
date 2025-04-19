import math
import heapq
import gymnasium as gym
from random import random
from collections import defaultdict
import numpy as np
import itertools
import csv
import statistics

class QLearner:
    def __init__(self, bucket_num, e, alpha, dsc, iter_change, e_change, alpha_change, dsc_change = 1, knowledge=None, vis=False):
        if vis:
            self.environment = gym.make('CartPole-v1', render_mode="human")
        else:
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
        if knowledge is not None:
            self.knowledge = knowledge
        else:
            self.knowledge = defaultdict(lambda:defaultdict(lambda: 0))
        self.vis = vis
        self.bucket_num = int(bucket_num)
        self.buckets = [np.linspace(self.lower_bounds[i], self.upper_bounds[i], self.bucket_num) for i in range(4)]
        self.e = e
        self.alpha = alpha
        self.dsc = dsc
        self.iter_change = iter_change
        self.e_change = e_change
        self.alpha_change = alpha_change
        self.dsc_change = dsc_change

    def learn(self, max_attempts, is_sarsa=False):
        rewards = []
        if not is_sarsa:
            for _ in range(1,max_attempts+1):
                reward_sum = self.attempt()

                if _%self.iter_change==0:
                    self.e = self.e*self.e_change
                    self.alpha = self.alpha*self.alpha_change
                    self.dsc = self.dsc*self.dsc_change
                rewards.append(reward_sum)
            return rewards
        else:
            for _ in range(1,max_attempts+1):
                reward_sum = self.attempt_sarsa()

                if _%self.iter_change==0:
                    self.e = self.e*self.e_change
                    self.alpha = self.alpha*self.alpha_change
                    self.dsc = self.dsc*self.dsc_change
                rewards.append(reward_sum)
            return rewards

    def attempt_sarsa(self):
        observation = self.discretise(self.environment.reset()[0])
        terminated, truncated  = False, False
        reward_sum = 0.0
        old_reward = 0
        old_action = 0
        while not truncated and not terminated:
            self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, terminated, truncated, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            self.update_knowledge_sarsa(old_action,action, observation ,new_observation, old_reward)
            observation = new_observation
            reward_sum += reward
            old_action = action
            old_reward = reward
        self.attempt_no += 1
        return reward_sum

    def update_knowledge_sarsa(self,old_action, action,old_observtion, observation,  old_reward):
        self.knowledge[old_observtion][old_action] = self.knowledge[old_observtion][old_action] + \
            self.alpha*(old_reward+self.dsc*self.knowledge[observation][action]-self.knowledge[old_observtion][old_action])



    def attempt(self):
        observation = self.discretise(self.environment.reset()[0])
        terminated, truncated  = False, False
        reward_sum = 0.0
        while not truncated and not terminated:
            self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, terminated, truncated, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def discretise(self, observation):
        res = []
        for i in range(4):
            bucket_ind = np.digitize(observation[i], self.buckets[i], right=True)
            res.append(bucket_ind)
        return tuple(res)

    def pick_action(self, observation):
        if not self.vis:
            if(random()<self.e):
                return self.environment.action_space.sample()
        if self.knowledge[observation][0] > self.knowledge[observation][1]:
            return 0
        return 1

    def update_knowledge(self, action, observation, new_observation, reward):
        self.knowledge[observation][action] = self.calculate_knowledge(action, observation, new_observation, reward)

    def calculate_knowledge(self, action, observation, new_observation, reward):
        best = 0
        if self.knowledge[new_observation][0] > self.knowledge[new_observation][1]:
            best = self.knowledge[new_observation][0]
        else:
            best = self.knowledge[new_observation][1]
        return (1-self.alpha)*self.knowledge[observation][action] + self.alpha*(reward + self.dsc*best)

def search_best_hyperparameters(n_best=5,is_sarsa=False):
    bucket_nums = [ 4, 5, 6, ]
    es = [0.1, 0.2, 0.5, 0.7]
    alphas = [ 0.1, 0.3, 0.6, 0.9]
    dscs = [0.1, 0.3, 0.6, 0.9]
    iter_changes = [100, 200]
    e_changes = [0.9, 0.8]
    alpha_changes = [0.9, 0.8]

    attempts_per_config = 3*max(iter_changes)

    best_params_heap = []
    for bucket_num, e, alpha, dsc, iter_change, e_change, alpha_change in itertools.product(bucket_nums, es, alphas, dscs, iter_changes, e_changes, alpha_changes):
        learner = QLearner(bucket_num, e, alpha, dsc, iter_change, e_change, alpha_change, 1)
        rewards = learner.learn(attempts_per_config, is_sarsa=is_sarsa)

        avg_reward = np.mean(rewards[-10:])

        print(f"Buckets: {bucket_num}, E: {e}, Alpha: {alpha}, Dsc: {dsc}, Iter_change: {iter_change}, E_change: {e_change}, Alpha_change: {alpha_change} => Avg Reward: {avg_reward:.2f}")

        config = (avg_reward, (bucket_num, e, alpha, dsc, iter_change, e_change, alpha_change))
        if len(best_params_heap) < n_best:
            heapq.heappush(best_params_heap, config)
        else:
            heapq.heappushpop(best_params_heap, config)

    best_params_sorted = sorted(best_params_heap, key=lambda x: -x[0])

    print("\nTop Hyperparameter Sets:")
    for i, (reward, params) in enumerate(best_params_sorted, 1):
        bucket_num, e, alpha, dsc, iter_change, e_change, alpha_change = params
        print(f"{i}. Buckets: {bucket_num}, E: {e}, Alpha: {alpha}, Dsc: {dsc}, "
              f"Iter_change: {iter_change}, E_change: {e_change}, "
              f"Alpha_change: {alpha_change} => Avg Reward: {reward:.2f}")

    return best_params_sorted

def train_and_evaluate(model_params, filename_base="res", iterations=10000, window_size=10, retries=2, is_sarsa=False):
    for k, params in enumerate(model_params):
        results_with_retries = []
        for _ in range(retries):
            learner = QLearner(*params)
            results = learner.learn(iterations, is_sarsa)
            results_mean = [statistics.mean(results[e-window_size:e]) for e in range(window_size, len(results))]
            results_with_retries.append(results_mean)
        params_str = ""
        for p in params:
            params_str+=str(p)+','
        with open("results/"+str(k)+filename_base+params_str+".csv", "w") as file:
            writer = csv.writer(file)
            for i, values in enumerate(zip(*results_with_retries)):
                mean = statistics.mean(values)
                stdev = statistics.stdev(values)
                writer.writerow((i, round(mean,2), round(stdev,2)))

search = True
evaluate = True
is_sarsa = False
def main():
    if search:
        params = search_best_hyperparameters(is_sarsa=is_sarsa)
        with open("params.csv", "w") as file:
            writer = csv.writer(file, delimiter=',')
            for row in params:
                writer.writerow(row[1])
    if evaluate:
        with open("params.csv", "r") as file:
            reader = csv.reader(file, delimiter=',')
            all_rows = []
            for row in reader:
                all_rows.append(tuple((float(r) for r in row)))
            train_and_evaluate(all_rows, iterations=4000, retries=3, is_sarsa=is_sarsa)

if __name__ == '__main__':
    main()
