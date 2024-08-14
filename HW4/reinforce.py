from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PiApproximationWithNN():
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        num_actions: the number of possible actions
        alpha: learning rate
        """
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self, s) -> int:
        s = torch.tensor(s, dtype=torch.float32)
        probs = self.model(s)
        action = torch.multinomial(probs, 1).item()  # Sample an action from the probability distribution
        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        gamma_t = torch.tensor(gamma_t, dtype=torch.float32)
        delta = torch.tensor(delta, dtype=torch.float32)

        self.optimizer.zero_grad()

        probs = self.model(s)
        log_prob = torch.log(probs[a])
        loss = -gamma_t * delta * log_prob
        loss.backward()

        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN():
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self, s) -> float:
        s = torch.tensor(s, dtype=torch.float32)
        value = self.model(s)
        return value.item()

    def update(self, s, G):
        """
        s: state S_t
        G: return G_t
        """
        s = torch.tensor(s, dtype=torch.float32)
        G = torch.tensor(G, dtype=torch.float32)

        self.optimizer.zero_grad()

        value = self.model(s)
        loss = nn.functional.mse_loss(value, G)
        loss.backward()

        self.optimizer.step()()


def REINFORCE(
    env,  # open-ai environment
    gamma: float,
    num_episodes: int,
    pi: PiApproximationWithNN,
    V: Baseline) -> Iterable[float]:
    """
    Implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    returns = []

    for episode in range(num_episodes):
        s = env.reset()
        done = False
        episode_log = []

        # Generate an episode
        while not done:
            a = pi(s)
            s_next, reward, done, _ = env.step(a)
            episode_log.append((s, a, reward))
            s = s_next

        G = 0
        for t in reversed(range(len(episode_log))):
            s, a, reward = episode_log[t]
            G = reward + gamma * G

            if isinstance(V, VApproximationWithNN):
                delta = G - V(s)
                V.update(s, G)
            else:
                delta = G - V(s)

            pi.update(s, a, gamma**t, delta)

        returns.append(episode_log[0][2])  # Store the return G_0

    return returns

