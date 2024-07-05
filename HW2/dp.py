from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env: EnvWithModel, pi: Policy, initV: np.array, theta: float) -> Tuple[np.array, np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """
    
    V = initV.copy()
    nS, nA = env.spec.nS, env.spec.nA
    gamma = env.spec.gamma
    
    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            new_value = 0
            for a in range(nA):
                action_prob = pi.action_prob(s, a)
                transition_value = 0
                for s_prime in range(nS):
                    transition_value += action_prob * env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
                new_value += transition_value
            V[s] = new_value
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                Q[s, a] += env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
                    
    return V, Q

def value_iteration(env: EnvWithModel, initV: np.array, theta: float) -> Tuple[np.array, Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    V = initV.copy()  # Initialize V(s) arbitrarily
    nS, nA = env.spec.nS, env.spec.nA
    gamma = env.spec.gamma

    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            # Compute the maximum expected value over all actions
            max_value = float('-inf')
            for a in range(nA):
                action_value = 0
                for s_prime in range(nS):
                    action_value += env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
                if action_value > max_value:
                    max_value = action_value
            V[s] = max_value
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break

    # Derive the optimal policy
    class DeterministicPolicy(Policy):
        def __init__(self, policy):
            self.policy = policy

        def action_prob(self, state: int, action: int) -> float:
            return 1.0 if self.policy[state] == action else 0.0

        def action(self, state: int) -> int:
            return self.policy[state]

    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        max_value = float('-inf')
        best_action = 0
        for a in range(nA):
            action_value = 0
            for s_prime in range(nS):
                action_value += env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
            if action_value > max_value:
                max_value = action_value
                best_action = a
        policy[s] = best_action

    return V, DeterministicPolicy(policy)
