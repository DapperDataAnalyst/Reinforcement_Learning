from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
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

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################
    
    V = initV.copy()  # Initialize V(s) arbitrarily
    V[-1] = 0  # Set V(terminal) to be zero
    nS, nA = env.spec.nS, env.spec.nA
    delta = 0
    
    while delta < theta:
        for s in range(nS):
            v = V[s]
            s_prime = s + 1 if s != nS - 1 else s
            new_value = 0
            for a in range(nA):
                new_value += pi.action_prob(s,a) * env.TD[s,a,s_prime] * (env.R[s,a,s_prime] + env.spec.gamma * V[s_prime])
            V[s] = new_value
            delta = max(delta, abs(v - V[s]))
        
    # Calculate Q(s, a) based on the stable V(s)
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            q_value = 0
            for prob, next_state, reward, done in env.P[s][a]:
                q_value += prob * (reward + V[next_state])
            Q[s, a] = q_value

    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    return V, pi
