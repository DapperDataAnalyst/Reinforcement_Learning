import numpy as np


class NonStationaryBandit:
    def __init__(self, k: int = 10):
        self.q_star = np.zeros(k)

    def reset(self):
        self.q_star = np.zeros_like(self.q_star)

    def step(self, a: int):
        """
        input:
            a: int, action
        return:
            reward: float, noisy reward of the selected bandit arm
            best_action: bool, whether the selected arm is the best arm
        """
        assert a < len(self.q_star)

        #####################
        # TODO: Implement the followings
        #  (1) determining whether the action a is the best action
        #  (2) generating noisy reward
        #  (3) each arm takes independent random walks
        #####################

        return reward, is_best_action


class ActionValue(object):
    def __init__(self, k: int, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.q = np.zeros(k)

    def reset(self):
        self.q = np.zeros(self.k)

    def update(self, a: int, r: float):
        raise NotImplementedError

    def epsilon_greedy_policy(self):
        """
        return:
            a: int, action
        """

        #####################
        # TODO: Implement the epsilon-greedy policy
        #####################

        return a


class SampleAverage(ActionValue):
    def __init__(self, k: int, epsilon: float):
        super().__init__(k, epsilon)
        self.n = np.zeros_like(self.q)

    def reset(self):
        self.n = np.zeros_like(self.q)

    def update(self, a: int, r: float):
        #####################
        # TODO: Implement how sample average method updates its ESTIMATED q value
        #####################


class ConstantStepSize(ActionValue):
    def __init__(self, alpha: float, k: int, epsilon: float):
        super().__init__(k, epsilon)
        self.alpha = alpha

    def update(self, a: int, r: float):
        #####################
        # TODO: Implement how constant step-size method updates its ESTIMATED q value
        #####################


def experiment(bandit: NonStationaryBandit, agent: ActionValue, steps: int):
    bandit.reset()
    agent.reset()

    rs = []
    best_action_taken = []

    for _ in range(steps):
        #####################
        # TODO: Implement how agent interacts with the bandit and updates its ESTIMATED q value
        #####################

        rs.append(r)
        best_action_taken.append(is_best_action)

    return np.array(rs), np.array(best_action_taken)


def main(seed: int = 0):
    np.random.seed(seed)
    N_bandit_runs = 300
    N_steps_for_each_bandit = 10000

    sample_average = SampleAverage(k=10, epsilon=0.1)
    constant = ConstantStepSize(k=10, epsilon=0.1, alpha=0.1)
    bandit = NonStationaryBandit()

    outputs = []

    for agent in [sample_average, constant]:
        average_rs, average_best_action_taken = [], []

        for i in range(N_bandit_runs):

            rs, best_action_taken = experiment(bandit, agent, N_steps_for_each_bandit)
            average_rs.append(rs)
            average_best_action_taken.append(best_action_taken)

        average_rs = np.mean(np.array(average_rs), axis=0)
        average_best_action_taken = np.mean(np.array(average_best_action_taken), axis=0)

        outputs += [average_rs, average_best_action_taken]

    return outputs[0], outputs[1], outputs[2], outputs[3]
