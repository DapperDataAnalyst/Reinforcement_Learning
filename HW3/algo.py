import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    for episode in range(num_episode):
        state = env.reset()
        states = [state]
        rewards = [0]
        T = float('inf')
        tau = 0
        t = 0
        
        while tau < T - 1:
            if t < T:
                action = pi.action(state)
                state, reward, done, info = env.step(action)
                states.append(state)
                rewards.append(reward)

                if done:
                    T = t + 1
            
            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma**(i-tau-1) * rewards[i] for i in range(tau+1, min(tau+n, T) + 1)])
                if tau + n < T:
                    G += gamma**n * V(states[tau + n])
                V.update(alpha, G, states[tau])
                
            t += 1
            if t < T:
                state = states[t]
    

