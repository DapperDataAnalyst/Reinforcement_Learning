import numpy as np
from tqdm import tqdm

from env import EnvSpec, Env, EnvWithModel
from policy import Policy

from dp import value_iteration, value_prediction
from monte_carlo import off_policy_mc_prediction_ordinary_importance_sampling as mc_ois
from monte_carlo import off_policy_mc_prediction_weighted_importance_sampling as mc_wis
from n_step_bootstrap import off_policy_n_step_sarsa as nsarsa
from n_step_bootstrap import on_policy_n_step_td as ntd

import argparse

## This is a random policy that takes a static probability distribution over all actions.
## If none is specified, all actions can be taken randomly.
class RandomPolicy(Policy):
    def __init__(self, nA, p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return np.random.choice(len(self.p), p=self.p)


LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

UPPER_LEFT = 0
UPPER_RIGHT = 1
LOWER_LEFT = 2
LOWER_RIGHT = 3

nS = 4
nA = 4

## This is a simple GridWorld environment.
## It is 2x2, so there are 4 possible states.
## 4 action: 0 left, 1 up, 2 right, 3 down
## You receive -1 rewards on all steps, until you hit the goal state on bottom right.
class GridWorld(EnvWithModel):  
    # GridWorld for example 4.1
    def __init__(self):
        # 4 states: 3 is terminal
        # 4 action: 0 left, 1 up, 2 right, 3 down
        env_spec = EnvSpec(nS, nA, 1.)
        super().__init__(env_spec)
        self.trans_mat, self.ret_mat = self._build_trans_mat()
        self.terminal_state = [LOWER_RIGHT]

    def _build_trans_mat(self):
        trans_mat = np.zeros((nS, nA, nS), dtype=int)
        ret_mat = -np.ones((nS, nA, nS))

        ## Setup the translation matrix, you can't go off the grid

        ## Upper left
        trans_mat[UPPER_LEFT][LEFT] [UPPER_LEFT] = 1
        trans_mat[UPPER_LEFT][UP]   [UPPER_LEFT] = 1
        trans_mat[UPPER_LEFT][RIGHT][UPPER_RIGHT] = 1
        trans_mat[UPPER_LEFT][DOWN] [LOWER_LEFT] = 1
        
        ## Upper right
        trans_mat[UPPER_RIGHT][LEFT] [UPPER_LEFT] = 1
        trans_mat[UPPER_RIGHT][UP]   [UPPER_RIGHT] = 1
        trans_mat[UPPER_RIGHT][RIGHT][UPPER_RIGHT] = 1
        trans_mat[UPPER_RIGHT][DOWN] [LOWER_RIGHT] = 1

        ## Bottom left
        trans_mat[LOWER_LEFT][LEFT] [LOWER_LEFT] = 1
        trans_mat[LOWER_LEFT][UP]   [UPPER_LEFT] = 1
        trans_mat[LOWER_LEFT][RIGHT][LOWER_RIGHT] = 1
        trans_mat[LOWER_LEFT][DOWN] [LOWER_LEFT] = 1

        ## Bottom right (terminal)
        for i in range(4):
            trans_mat[LOWER_RIGHT][i][LOWER_RIGHT] = 1
            trans_mat[LOWER_RIGHT][i][LOWER_RIGHT] = 1
            trans_mat[LOWER_RIGHT][i][LOWER_RIGHT] = 1
            trans_mat[LOWER_RIGHT][i][LOWER_RIGHT] = 1

            ret_mat[LOWER_RIGHT][i][LOWER_RIGHT] = 0

        return trans_mat, ret_mat

    @property
    def TD(self):
        return self.trans_mat

    @property
    def R(self):
        return self.ret_mat

    def reset(self):
        # Random initialze location for each episode run
        self.state = np.random.randint(0, 2) # (not inclusive)
        return self.state

    def step(self, action):
        assert action in range(self.spec.nA), "Invalid Action"
        assert self.state not in self.terminal_state, "Episode has ended!"

        prev_state = self.state
        self.state = np.random.choice(self.spec.nS, p=self.trans_mat[self.state, action])
        r = self.ret_mat[prev_state, action, self.state]

        if self.state in self.terminal_state:
            return self.state, r, True
        else:
            return self.state, r, False


def visualize(pi):
    # Visulize policy with some strings
    visual_policy = np.empty(4, dtype=object)
    viz_map = {
        LEFT: '←',
        RIGHT: '→',
        UP: '↑',
        DOWN: '↓'
    }
    for s in range(nA):
        visual_policy[s] = viz_map[pi.action(s)]
    return visual_policy

def Q2V(Q, pi):
    # Compute V based on Q and policy pi
    V = np.zeros(4)
    for s in range(4):
        V[s] = 0
        for a in range(4):
            V[s] += pi.action_prob(s, a) * Q[s, a]
    return V

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run more advanced tests.")

    parser.add_argument('--dp', action='store_true', help='Test dp.py')
    parser.add_argument('--mc', action='store_true', help='Test monte_carlo.py')
    parser.add_argument('--n_step', action='store_true', help='Test n_step_bootstrap.py')

    args = parser.parse_args()

    if (not args.dp and not args.mc and not args.n_step):
        args.dp = True
        args.mc = True
        args.n_step = True

    nA = 4
    nS = 4
    grid_world = GridWorld()
    behavior_policy = RandomPolicy(nA)
    initV = np.zeros(nS)

    if (args.dp):
        print("DP value prediction under random policy")
        V, Q = value_prediction(grid_world, behavior_policy, initV, 1e-12)
        print(V.reshape((2, 2)))
        print()

        print("DP value iteration optimal value and policy")
        V, pi = value_iteration(grid_world, initV, 1e-12)
        print(V.reshape((2, 2)))
        print(visualize(pi).reshape((2, 2)))
        print()

    if (args.mc or args.n_step):
        # Sample with random policy
        N_EPISODES = 10000

        print("Generating episodes based on random policy")
        trajs = []
        for _ in tqdm(range(N_EPISODES)):
            s = grid_world.reset()
            traj = []

            while s != 3:
                a = behavior_policy.action(s)
                next_s, r, _ = grid_world.step(a)
                traj.append((s, a, r, next_s))
                s = next_s
            trajs.append(traj)

        if args.mc:
            # On-policy evaluation tests for random policy
            # OIS
            Q_est_ois = mc_ois(grid_world.spec, trajs, behavior_policy, behavior_policy,
                            np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
            
            print("On random policy value OIS: ")
            print(Q2V(Q_est_ois, behavior_policy).reshape((2, 2)))
            print()

            # WIS
            Q_est_wis = mc_wis(grid_world.spec, trajs, behavior_policy, behavior_policy,
                            np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
        
            print("On random policy value WIS: ")
            print(Q2V(Q_est_wis, behavior_policy).reshape((2, 2)))
            print()

            # Off-policy evaluation test with optimal policy
            class OptimalPolicy(Policy):
                def action_prob(self, state, action):
                    return 1.0 if self.action(state) == action else 0.0

                def action(self, state):
                    return [RIGHT, DOWN, RIGHT, LEFT][state]

            #V, pi_star = value_iteration(grid_world, initV, 1e-12)
            pi_star = OptimalPolicy()

            Q_est_ois = mc_ois(grid_world.spec, trajs, behavior_policy, pi_star, np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
            print("Off policy evaluation for optimal value OIS: ")
            print(Q2V(Q_est_ois, pi_star).reshape((2, 2)))
            print()

            Q_est_wis = mc_wis(grid_world.spec, trajs, behavior_policy, pi_star, np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
            print("Off policy evaluation for optimal value WIS: ")
            print(Q2V(Q_est_wis, pi_star).reshape((2, 2)))
            print()

        if args.n_step:
            for n in range(1, 2):
                # n-step TD with alpha = 0.005
                V_est_td = ntd(grid_world.spec, trajs, n, 0.005, np.zeros((grid_world.spec.nS)))
                print(f"{n}-step TD value estimation on random policy: ")
                print(V_est_td.reshape((2, 2)))
                print()

                # Off-policy SARSA
                # n-step with alpha = 0.01, should converge to v*
                Q_star_est, pi_star_est = nsarsa(grid_world.spec, trajs, behavior_policy, n=n, alpha=0.01,
                                                initQ=np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
                print(f"{n}-step SARSA off policy optimal value est. :")
                print(Q2V(Q_star_est, pi_star_est).reshape((2, 2)))
                print()
                print(f"{n}-step SARSA off policy optimal policy :")
                print(visualize(pi_star_est).reshape((2, 2)))
                print()