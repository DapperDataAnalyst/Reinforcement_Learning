from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    bpi: Policy,
    pi: Policy,
    initQ: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    Q = initQ.copy()
    C = np.zeros_like(Q)  # Initialize C(s, a) = 0

    gamma = env_spec.gamma
    
    for episode in trajs:
        G = 0
        W = 1
        
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t1, s_t1 = episode[t]
            G = gamma * G + r_t1
            C[s_t, a_t] += W
            Q[s_t, a_t] += (W / C[s_t, a_t]) * (G - Q[s_t, a_t])
            
            # Update W
            if pi.action_prob(s_t, a_t) == 0:
                break
            W *= pi.action_prob(s_t, a_t) / bpi.action_prob(s_t, a_t)
            if W == 0:
                break

    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    bpi: Policy,
    pi: Policy,
    initQ: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    Q = initQ.copy()
    C = np.zeros_like(Q)  # Initialize C(s, a) = 0

    gamma = env_spec.gamma
    
    for episode in trajs:
        G = 0
        W = 1
        
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t1, s_t1 = episode[t]
            G = gamma * G + r_t1
            C[s_t, a_t] += W
            Q[s_t, a_t] += (W / C[s_t, a_t]) * (G - Q[s_t, a_t])
            
            # Update W
            if pi.action_prob(s_t, a_t) == 0:
                break
            W *= pi.action_prob(s_t, a_t) / bpi.action_prob(s_t, a_t)
            if W == 0:
                break

    return Q
