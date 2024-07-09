from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    n: int,
    alpha: float,
    initV: np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t, a_t, r_{t+1}, s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_\pi$ function; numpy array shape of [nS]
    """

    V = initV.copy()
    gamma = env_spec.gamma

    for episode in trajs:
        T = len(episode)
        states = [step[0] for step in episode] #+ [episode[-1][3]]  # S_0 to S_T
        rewards = [step[2] for step in episode]  # R_1 to R_T

        for t in range(T):
            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma**(i - tau) * rewards[i] for i in range(tau, min(tau + n, T)))
                if tau + n < T:
                    G += gamma**n * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])

    return V


def off_policy_n_step_sarsa(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    bpi: Policy,
    n: int,
    alpha: float,
    initQ: np.array
) -> Tuple[np.array, Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t, a_t, r_{t+1}, s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS, nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS, nA]
        policy: $pi_star$; instance of policy class
    """

    Q = initQ.copy()
    gamma = env_spec.gamma
    nS, nA = env_spec.nS, env_spec.nA

    def greedy_policy(Q):
        class GreedyPolicy(Policy):
            def action_prob(self, state: int, action: int) -> float:
                return 1.0 if action == np.argmax(Q[state]) else 0.0

            def action(self, state: int) -> int:
                return np.argmax(Q[state])
        
        return GreedyPolicy()

    pi = greedy_policy(Q)  # Initial policy is greedy with respect to Q

    for episode in trajs:
        T = len(episode)
        states = [step[0] for step in episode] #+ [episode[-1][3]]  # S_0 to S_T
        actions = [step[1] for step in episode] + [0]  # A_0 to A_T, add dummy action at the end
        rewards = [step[2] for step in episode]  # R_1 to R_T

        for t in range(T):
            # s_t, a_t, r_t1, s_t1 = episode[t]
            # actions[t] = bpi.action(s_t)
            # actions[t + 1] = bpi.action(s_t1)

            tau = t - n + 1
            if tau >= 0:
                rho = np.prod([pi.action_prob(states[i], actions[i]) / bpi.action_prob(states[i], actions[i]) for i in range(tau, min(tau + n, T))])
                G = sum(gamma**(i - tau) * rewards[i] for i in range(tau, min(tau + n, T)))
                if tau + n < T:
                    G += gamma**n * Q[states[tau + n], actions[tau + n]]
                Q[states[tau], actions[tau]] += alpha * rho * (G - Q[states[tau], actions[tau]])

                # Update policy to be greedy with respect to Q
                pi = greedy_policy(Q)

    return Q, pi
